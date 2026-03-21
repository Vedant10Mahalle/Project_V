from flask import Flask, render_template, request, session, redirect, url_for, flash
from werkzeug.security import check_password_hash
import pandas as pd
import os
import pickle

# Load the trained RandomForest model if available
_ML_MODEL = None
try:
    with open('models/risk_model.pkl', 'rb') as _f:
        _ML_MODEL = pickle.load(_f)
    print("INFO: RandomForest ML model loaded successfully from models/risk_model.pkl")
except FileNotFoundError:
    print("WARN: No trained model found. Using deterministic risk engine fallback.")

app = Flask(__name__)
app.secret_key = "super_secret_nmam_key"

# ---------- Data Loaders ----------
def get_df(filepath):
    if not os.path.exists(filepath): return pd.DataFrame()
    try: return pd.read_csv(filepath)
    except pd.errors.EmptyDataError: return pd.DataFrame()

def save_df(df, filepath):
    df.to_csv(filepath, index=False)

def load_teacher_dashboard_data(teacher_id):
    students_df = get_df("data/students.csv")
    if students_df.empty: return []
    my_students = students_df[students_df['mentor_id'] == teacher_id].copy()
    if my_students.empty: return []
        
    academic_df = get_df("data/academic_summary.csv")
    if not academic_df.empty: 
        # Get the latest semester for each student to highlight in dashboard
        latest_acad = academic_df.sort_values('semester').groupby('usn').last().reset_index()
        my_students = pd.merge(my_students, latest_acad, on="usn", how="left")
        
    my_students.fillna({'attendance': 0, 'sgpa': 0, 'activity_points': 0, 'semester_y': my_students['semester_x']}, inplace=True)
    
    def calc_year(sem):
        try: return f"{(int(float(sem)) - 1) // 2 + 1} Year"
        except: return "1 Year"
    my_students['year'] = my_students['semester_x'].apply(calc_year)
    
    records = my_students.to_dict(orient="records")
    for r in records:
        prof = calculate_multi_factor_risk(r)
        r['acad_risk'] = prof['academic_risk']
        r['behav_risk'] = prof['behavioral_risk']
        r['final_risk'] = prof['level']
        r['dropout_prob'] = prof['dropout_prob']
        
    return records

def load_student_profile(usn):
    students_df = get_df("data/students.csv")
    if students_df.empty: return {}
    me = students_df[students_df['usn'] == usn]
    if me.empty: return {}
    record = me.iloc[0].to_dict()
    
    # Mentor info
    teachers_df = get_df("data/teachers.csv")
    if not teachers_df.empty and pd.notna(record.get('mentor_id')):
        mentor = teachers_df[teachers_df['teacher_id'] == record['mentor_id']]
        if not mentor.empty: record['mentor_name'] = mentor.iloc[0]['name']

    # Academic Summaries (All semantics)
    summ_df = get_df("data/academic_summary.csv")
    record['semesters'] = []
    record['sem_summaries'] = {}
    
    # Initialize defaults
    record['attendance'] = 0
    record['sgpa'] = 0.0
    record['cgpa'] = 0.0
    record['activity_points'] = 0

    if not summ_df.empty and 'usn' in summ_df.columns:
        my_summs = summ_df[summ_df['usn'] == usn]
        record['semesters'] = my_summs.to_dict(orient='records')
        
        for s in record['semesters']:
            record['sem_summaries'][str(int(s['semester']))] = s
            
        # Populate the latest baseline metrics automatically
        record['attendance'] = 0
        record['sgpa'] = 0.0
        record['cgpa'] = 0.0
        record['activity_points'] = 0
        
        if len(record['semesters']) > 0:
            latest = record['semesters'][-1]
            record['attendance'] = latest.get('attendance', 0)
            record['sgpa'] = latest.get('sgpa', 0.0)
            record['cgpa'] = latest.get('cgpa', 0.0)
            record['activity_points'] = latest.get('activity_points', 0)

    # Detailed Marks Grouped by Semester
    marks_df = get_df("data/academic_marks.csv")
    record['marks'] = {} # { semester_number: [ {subject: x, cie: y, see: z} ] }
    if not marks_df.empty and 'usn' in marks_df.columns:
        my_marks = marks_df[marks_df['usn'] == usn]
        for _, row in my_marks.iterrows():
            sem = str(int(row['semester'])) if pd.notna(row['semester']) else '0'
            if sem not in record['marks']: record['marks'][sem] = []
            record['marks'][sem].append({
                'subject': row['subject_name'],
                'cie': row['cie'],
                'see': row['see']
            })
            
    return record

def calculate_multi_factor_risk(student_record):
    """Hybrid Intelligence System calculating Academic, Behavioral, and Final Risk with NLP Reason Generation."""
    feed_df = get_df("data/weekly_feedback.csv")
    recent_stress = 0
    recent_issues = "no"
    und_lvl = 3
    miss_cls = 0
    nd_help = "no"
    my_feed_len = 0
    
    if not feed_df.empty and 'usn' in feed_df:
        my_feed = feed_df[feed_df['usn'] == student_record.get('usn')]
        my_feed_len = len(my_feed)
        if not my_feed.empty:
            last_feed = my_feed.iloc[-1]
            try: recent_stress = float(last_feed.get('stress', 0))
            except: recent_stress = 0
            recent_issues = str(last_feed.get('academic_issues', 'no')).lower()
            try: und_lvl = int(last_feed.get('understanding_level', 3))
            except: und_lvl = 3
            try: miss_cls = int(last_feed.get('missed_classes', 0))
            except: miss_cls = 0
            nd_help = str(last_feed.get('need_help', 'no')).lower()

    # 1. Behavioral Scoring
    beh_score = 0
    if recent_stress >= 7: beh_score += 2
    elif recent_stress >= 4: beh_score += 1
    if recent_issues == "yes": beh_score += 1
    if und_lvl <= 2: beh_score += 2
    if miss_cls >= 2: beh_score += 1
    if nd_help == 'yes': beh_score += 2
    beh_level = "High" if beh_score >= 4 else ("Medium" if beh_score >= 2 else "Low")
    
    # 2. Academic Scoring
    acad_score = 0
    att = float(student_record.get('attendance', 100))
    sgpa = float(student_record.get('sgpa', 10))
    if pd.isna(sgpa) or sgpa == 0: sgpa = 10 
    
    if att < 75: acad_score += 3
    elif att < 85: acad_score += 1
    if sgpa < 5.0: acad_score += 3
    elif sgpa < 7.0: acad_score += 1
    acad_level = "High" if acad_score >= 3 else ("Medium" if acad_score >= 1 else "Low")
    
    # 3. Final Risk Engine
    if acad_level == "High" or beh_level == "High": final_level = "High"
    elif acad_level == "Medium" and beh_level == "Medium": final_level = "Medium"
    elif acad_level == "Medium" or beh_level == "Medium": final_level = "Medium"
    else: final_level = "Low"
    
    # 4. Reason Generation (Explainability)
    reasons = []
    if att < 85: reasons.append(f"Low attendance ({att}%)")
    if sgpa < 7.0: reasons.append(f"Below average performance (SGPA: {sgpa})")
    if recent_stress >= 7: reasons.append(f"High stress logged in weekly feedback")
    if recent_issues == "yes": reasons.append("Reported academic difficulties recently")
    if und_lvl <= 2: reasons.append(f"Low self-reported understanding level ({und_lvl}/5)")
    if miss_cls >= 2: reasons.append(f"High number of recently missed classes ({miss_cls})")
    if nd_help == 'yes': reasons.append("Student explicitly requested faculty intervention")
    if not reasons: reasons.append("Consistent performance with stable engagement")
    
    # 5. Probability Extraction — uses RandomForest ML model if available
    if _ML_MODEL is not None:
        try:
            import numpy as np
            features = [[att, sgpa, recent_stress, miss_cls]]
            ml_prob = _ML_MODEL.predict_proba(features)[0][1]  # probability of high-risk class
            dropout_prob = round(min(98, max(2, ml_prob * 100)))
        except Exception:
            dropout_prob = round(min(98, 10 + (acad_score * 10) + (beh_score * 8)))
    else:
        base_prob = 10
        if final_level == "High": base_prob = 75 + (acad_score * 3) + (beh_score * 3)
        elif final_level == "Medium": base_prob = 40 + (acad_score * 3) + (beh_score * 3)
        else: base_prob = 5 + (acad_score * 3)
        dropout_prob = round(min(98, base_prob))

    confidence_score = round(min(0.95, 0.70 + (0.02 * my_feed_len)), 2)
    
    return {
        "score": dropout_prob, 
        "academic_risk": acad_level,
        "behavioral_risk": beh_level,
        "level": final_level,
        "dropout_prob": dropout_prob,
        "confidence": confidence_score,
        "reasons": reasons,
        "stress": recent_stress
    }

# ---------- Routing & Auth ----------

@app.route("/")
def home():
    if "user_id" not in session: return redirect(url_for("login"))
    role = session.get("role")
    if role == "teacher": return redirect(url_for("teacher"))
    elif role == "student":
        if session.get("needs_mentor"): return redirect(url_for("select_mentor"))
        return redirect(url_for("student"))
    elif role == "parent": return redirect(url_for("parent"))
    elif role == "admin": return redirect(url_for("admin_dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        role = request.form.get("role")
        username = request.form.get("username")
        password = request.form.get("password")
        
        def _verify(stored, entered):
            stored = str(stored)
            # Support both hashed (werkzeug) and legacy plaintext passwords
            if stored.startswith('scrypt:') or stored.startswith('pbkdf2:'):
                return check_password_hash(stored, entered)
            return stored == entered

        if role == "teacher":
            df = get_df("data/teachers.csv")
            matching = df[df["teacher_id"] == username]
            if not matching.empty and _verify(matching.iloc[0]["password"], password):
                session["user_id"] = matching.iloc[0]["teacher_id"]
                session["name"] = matching.iloc[0]["name"]
                session["role"] = "teacher"
                return redirect(url_for("home"))

        elif role == "student":
            df = get_df("data/students.csv")
            matching = df[df["usn"] == username]
            if not matching.empty and _verify(matching.iloc[0]["password"], password):
                session["user_id"] = matching.iloc[0]["usn"]
                session["name"] = matching.iloc[0]["name"]
                session["role"] = "student"
                mentor_val = matching.iloc[0]["mentor_id"]
                session["needs_mentor"] = pd.isna(mentor_val) or str(mentor_val).strip() == ""
                if not session["needs_mentor"]: session["mentor_id"] = mentor_val
                return redirect(url_for("home"))

        elif role == "parent":
            df = get_df("data/parents.csv")
            matching = df[df["parent_id"] == username]
            if not matching.empty and _verify(matching.iloc[0]["password"], password):
                session["user_id"] = matching.iloc[0]["parent_id"]
                session["role"] = "parent"
                session["related_usn"] = matching.iloc[0]["usn"]
                return redirect(url_for("home"))
                
        elif role == "admin":
            df = get_df("data/admins.csv")
            matching = df[df["admin_id"] == username]
            if not matching.empty and _verify(matching.iloc[0]["password"], password):
                session["user_id"] = matching.iloc[0]["admin_id"]
                session["name"] = matching.iloc[0]["name"]
                session["role"] = "admin"
                return redirect(url_for("home"))

        flash("Invalid credentials or role mismatch.")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        usn = request.form.get("usn").strip()
        name = request.form.get("name").strip()
        branch = request.form.get("branch").strip()
        semester = request.form.get("semester").strip()
        password = request.form.get("password")
        parent_name = request.form.get("parent_name", "").strip()
        parent_phone = request.form.get("parent_phone", "").strip()
        
        df = get_df("data/students.csv")
        if not df.empty and usn in df['usn'].values:
            flash("USN already registered. Please log in.")
            return redirect(url_for("register"))
            
        from werkzeug.security import generate_password_hash
        new_row = {'usn': usn, 'name': name, 'branch': branch, 'semester': semester, 'mentor_id': '', 'password': generate_password_hash(password)}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        save_df(df, "data/students.csv")
        
        # Auto-create Parent
        if parent_name and parent_phone:
            pdf = get_df("data/parents.csv")
            parent_id = f"P-{usn}"
            if pdf.empty or parent_id not in pdf['parent_id'].values:
                new_parent = {'parent_id': parent_id, 'name': parent_name, 'usn': usn, 'password': generate_password_hash("welcome123")}
                pdf = pd.concat([pdf, pd.DataFrame([new_parent])], ignore_index=True)
                save_df(pdf, "data/parents.csv")
            
            flash(f"Registration successful! SMS sent to {parent_phone} with Parent Login ID: {parent_id} and Password: welcome123")
        else:
            flash("Registration successful! Please log in to select your Mentor.")
            
        return redirect(url_for("login"))
        
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/student/select_mentor", methods=["GET", "POST"])
def select_mentor():
    if session.get("role") != "student": return redirect(url_for("login"))
    if not session.get("needs_mentor"): return redirect(url_for("home"))
    
    if request.method == "POST":
        chosen_mentor = request.form.get("mentor_id")
        usn = session.get("user_id")
        df = get_df("data/students.csv")
        idx = df[df['usn'] == usn].index[0]
        df.loc[idx, 'mentor_id'] = chosen_mentor
        save_df(df, "data/students.csv")
        session["needs_mentor"] = False
        session["mentor_id"] = chosen_mentor
        return redirect(url_for("home"))
        
    teachers = get_df("data/teachers.csv").to_dict(orient="records")
    return render_template("select_mentor.html", teachers=teachers)

# ---------- Dashboards & Entry Forms ----------

@app.route("/teacher")
def teacher():
    if session.get("role") != "teacher": return redirect(url_for("login"))
    user_id = session.get("user_id")
    students_list = load_teacher_dashboard_data(user_id)
    
    meet_df = get_df("data/meeting_requests.csv")
    meetings = []
    if not meet_df.empty and "teacher_id" in meet_df.columns:
        my_meets = meet_df[(meet_df["teacher_id"] == user_id) & (meet_df["status"] == "Pending")]
        for _, m in my_meets.iterrows():
            prof = load_student_profile(m['usn'])
            meetings.append({"usn": m['usn'], "student_name": prof.get('name', 'Unknown')})
            
    return render_template("teacher.html", students=students_list, name=session.get("name"), meetings=meetings)

@app.route("/teacher/resolve_meeting/<usn>")
def resolve_meeting(usn):
    if session.get("role") != "teacher": return redirect(url_for("login"))
    df = get_df("data/meeting_requests.csv")
    if not df.empty:
        idx = df[(df["teacher_id"] == session.get("user_id")) & (df["usn"] == usn) & (df["status"] == "Pending")].index
        if len(idx) > 0:
            df.loc[idx[0], "status"] = "Resolved"
            save_df(df, "data/meeting_requests.csv")
    return redirect(url_for("teacher"))

@app.route("/teacher/student/<usn>")
def teacher_student_view(usn):
    if session.get("role") != "teacher": return redirect(url_for("login"))
    
    profile = load_student_profile(usn)
    if not profile or profile.get('mentor_id') != session.get('user_id'):
        return "Unauthorized Access. Student not in your batch.", 403
        
    risk_profile = calculate_multi_factor_risk(profile)
    
    feed_df = get_df("data/weekly_feedback.csv")
    history = []
    if not feed_df.empty and 'usn' in feed_df.columns:
        history = feed_df[feed_df['usn'] == usn].to_dict(orient="records")
        
    return render_template("teacher_student_view.html", student=profile, risk=risk_profile, history=history)

@app.route("/teacher/student/<usn>/report")
def teacher_student_report(usn):
    if session.get("role") != "teacher": return redirect(url_for("login"))
    
    profile = load_student_profile(usn)
    if not profile or profile.get('mentor_id') != session.get('user_id'):
        return "Unauthorized Access.", 403
        
    risk_profile = calculate_multi_factor_risk(profile)
    feed_df = get_df("data/weekly_feedback.csv")
    history = feed_df[feed_df['usn'] == usn].to_dict(orient="records") if not feed_df.empty and 'usn' in feed_df.columns else []
        
    return render_template("report.html", student=profile, risk=risk_profile, history=history)

@app.route("/teacher/notify_students", methods=["POST"])
def notify_students():
    if session.get("role") != "teacher": return redirect(url_for("login"))
    teacher_id = session.get("user_id")
    df = get_df("data/notifications.csv")
    if df.empty: df = pd.DataFrame(columns=["teacher_id", "active"])
    idx = df[df['teacher_id'] == teacher_id].index
    if len(idx) > 0: df.loc[idx[0], "active"] = "True"
    else: df = pd.concat([df, pd.DataFrame([{"teacher_id": teacher_id, "active": "True"}])], ignore_index=True)
    save_df(df, "data/notifications.csv")
    flash("Reminder dispatched to all assigned students: 'Fill weekly form'.")
    return redirect(url_for("teacher"))

@app.route("/student")
def student():
    if session.get("role") != "student": return redirect(url_for("login"))
    if session.get("needs_mentor"): return redirect(url_for("select_mentor"))
    
    profile = load_student_profile(session.get("user_id"))
    
    # Check notifications
    df = get_df("data/notifications.csv")
    active_alert = False
    if not df.empty and profile.get("mentor_id"):
        notifs = df[df['teacher_id'] == profile["mentor_id"]]
        if not notifs.empty and str(notifs.iloc[0]["active"]) == "True":
            active_alert = True
            
    return render_template("student.html", student=profile, active_alert=active_alert)

@app.route("/student/dismiss_alert")
def dismiss_alert():
    teacher_id = session.get("mentor_id")
    df = get_df("data/notifications.csv")
    if not df.empty:
        idx = df[df['teacher_id'] == teacher_id].index
        if len(idx) > 0: df.loc[idx[0], "active"] = "False"
        save_df(df, "data/notifications.csv")
    return redirect(url_for("student"))

@app.route("/student/academic_entry", methods=["GET", "POST"])
def student_academic_entry():
    if session.get("role") != "student": return redirect(url_for("login"))
    usn = session.get("user_id")
    
    if request.method == "POST":
        sem = request.form.get("semester")
        
        # 1. Save Summary
        summ_df = get_df("data/academic_summary.csv")
        if summ_df.empty: summ_df = pd.DataFrame(columns=['usn','semester','attendance','sgpa','cgpa','activity_points'])
        
        summ_data = {
            'usn': usn, 'semester': sem,
            'attendance': request.form.get("attendance"),
            'sgpa': request.form.get("sgpa"),
            'cgpa': request.form.get("cgpa"),
            'activity_points': request.form.get("activity_points")
        }
        
        # Replace if exists
        summ_df = summ_df[~((summ_df['usn'] == usn) & (summ_df['semester'].astype(str) == str(sem)))]
        summ_df = pd.concat([summ_df, pd.DataFrame([summ_data])], ignore_index=True)
        save_df(summ_df, "data/academic_summary.csv")
        
        # 2. Save Dynamic Marks
        marks_df = get_df("data/academic_marks.csv")
        if marks_df.empty: marks_df = pd.DataFrame(columns=['usn','semester','subject_name','cie','see'])
        
        subs = request.form.getlist("subject_name[]")
        cies = request.form.getlist("cie[]")
        sees = request.form.getlist("see[]")
        
        marks_df = marks_df[~((marks_df['usn'] == usn) & (marks_df['semester'].astype(str) == str(sem)))]
        
        new_marks = []
        for i in range(len(subs)):
            if subs[i].strip() != "":
                new_marks.append({'usn': usn, 'semester': sem, 'subject_name': subs[i], 'cie': cies[i] if cies[i] else 0, 'see': sees[i] if sees[i] else 0})
                
        if new_marks:
            marks_df = pd.concat([marks_df, pd.DataFrame(new_marks)], ignore_index=True)
            
        save_df(marks_df, "data/academic_marks.csv")
        return redirect(url_for("student"))
        
    return render_template("student_academic_entry.html")

@app.route("/student/weekly_checkin", methods=["GET", "POST"])
def student_checkin():
    if session.get("role") != "student": return redirect(url_for("login"))
    usn = session.get("user_id")
    
    file = "data/weekly_feedback.csv"
    df = get_df(file)
    if df.empty: df = pd.DataFrame(columns=['usn','week','stress','academic_issues','personal_issues'])
    
    if request.method == "POST":
        week = request.form.get("week")
        stress = request.form.get("stress")
        ac_issue = request.form.get("academic_issues")
        per_issue = request.form.get("personal_issues")
        und_lvl = request.form.get("understanding_level", "3")
        miss_cls = request.form.get("missed_classes", "0")
        nd_help = request.form.get("need_help", "No")
        
        df = pd.concat([df, pd.DataFrame([{
            'usn': usn, 'week': week, 'stress': stress, 
            'academic_issues': ac_issue, 'personal_issues': per_issue,
            'understanding_level': und_lvl, 'missed_classes': miss_cls, 'need_help': nd_help
        }])], ignore_index=True)
        save_df(df, file)
        return redirect(url_for("student"))
        
    return render_template("student_weekly_checkin.html")

# ---------- Phase 3: Parent Route ----------
@app.route("/parent")
def parent():
    if session.get("role") != "parent": return redirect(url_for("login"))
    
    usn = session.get("related_usn")
    profile = load_student_profile(usn)
    if not profile: return "Error loading student profile."
    
    risk_profile = calculate_multi_factor_risk(profile)
    
    return render_template("parent.html", student=profile, risk=risk_profile)

@app.route("/parent/request_meeting", methods=["POST"])
def request_meeting():
    if session.get("role") != "parent": return redirect(url_for("login"))
    parent_id = session.get("user_id")
    usn = session.get("related_usn")
    
    profile = load_student_profile(usn)
    teacher_id = profile.get("mentor_id")
    if pd.isna(teacher_id) or not teacher_id:
        flash("Cannot request a meeting: Student has no formally assigned mentor yet.")
        return redirect(url_for("parent"))
        
    df = get_df("data/meeting_requests.csv")
    if df.empty: df = pd.DataFrame(columns=["parent_id", "teacher_id", "usn", "status"])
    
    existing = df[(df["parent_id"] == parent_id) & (df["status"] == "Pending")]
    if not existing.empty:
        flash("You already have an active pending meeting request with this mentor.")
        return redirect(url_for("parent"))
        
    new_req = {"parent_id": parent_id, "teacher_id": teacher_id, "usn": usn, "status": "Pending"}
    df = pd.concat([df, pd.DataFrame([new_req])], ignore_index=True)
    save_df(df, "data/meeting_requests.csv")
    
    flash("Formal Meeting Request dispatched to the assigned Faculty Mentor.")
    return redirect(url_for("parent"))

# ═══════════════════════════════════════════════════════
# ADMIN ROUTES & FORGOT PASSWORD
# ═══════════════════════════════════════════════════════

@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        role = request.form.get("role")
        id_num = request.form.get("id_num").strip()
        name = request.form.get("name").strip().lower()
        new_password = request.form.get("new_password")
        from werkzeug.security import generate_password_hash
        
        if role == "teacher": df = get_df("data/teachers.csv"); id_col = "teacher_id"
        elif role == "student": df = get_df("data/students.csv"); id_col = "usn"
        elif role == "parent": df = get_df("data/parents.csv"); id_col = "parent_id"
        else: return redirect(url_for("forgot_password"))

        if not df.empty and id_col in df.columns:
            idx = df[(df[id_col] == id_num) & (df["name"].str.lower() == name)].index
            if len(idx) > 0:
                df.loc[idx[0], "password"] = generate_password_hash(new_password)
                save_df(df, f"data/{role}s.csv")
                flash("Password updated successfully! You can now log in.", "success")
                return redirect(url_for("login"))
            
        flash("Could not verify your identity. Check your ID and Name.", "error")
        return redirect(url_for("forgot_password"))

    return render_template("forgot_password.html")

import io
@app.route("/admin/upload/<entity>", methods=["POST"])
def admin_upload_csv(entity):
    if not admin_required(): return redirect(url_for("login"))
    if 'file' not in request.files:
        flash("No file provided.", "error")
        return redirect(url_for("admin_dashboard"))
    file = request.files['file']
    if file.filename == '':
        flash("No selected file.", "error")
        return redirect(url_for("admin_dashboard"))
        
    try:
        from werkzeug.security import generate_password_hash
        default_pwd = generate_password_hash("welcome123")
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        uploaded_df = pd.read_csv(stream)
        
        if entity == "teachers":
            target = "data/teachers.csv"
            df = get_df(target)
            dept = request.form.get("department", "General")
            for _, row in uploaded_df.iterrows():
                tid = str(row.get("teacher_id")).strip()
                name = str(row.get("name")).strip()
                if not df.empty and tid in df["teacher_id"].values: continue
                new_row = {"teacher_id": tid, "name": name, "department": dept, "password": default_pwd}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_df(df, target)
            
        elif entity == "students":
            target = "data/students.csv"
            df = get_df(target)
            branch = request.form.get("branch", "General")
            for _, row in uploaded_df.iterrows():
                usn = str(row.get("usn")).strip()
                name = str(row.get("name")).strip()
                sem = str(row.get("semester", "1")).strip()
                if not df.empty and usn in df["usn"].values: continue
                new_row = {"usn": usn, "name": name, "branch": branch, "semester": sem, "mentor_id": "", "password": default_pwd}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_df(df, target)
            
        elif entity == "parents":
            target = "data/parents.csv"
            df = get_df(target)
            for _, row in uploaded_df.iterrows():
                pid = str(row.get("parent_id")).strip()
                name = str(row.get("name")).strip()
                usn = str(row.get("usn")).strip()
                if not df.empty and pid in df["parent_id"].values: continue
                new_row = {"parent_id": pid, "name": name, "usn": usn, "password": default_pwd}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_df(df, target)
            
        flash(f"Successfully uploaded and processed {entity} CSV.", "success")
    except Exception as e:
        flash(f"Error processing CSV: {str(e)}", "error")
        
    return redirect(url_for("admin_dashboard"))


def admin_required():
    """Returns True if access is OK, False otherwise."""
    return session.get("role") == "admin"

@app.route("/admin")
def admin_dashboard():
    if not admin_required(): return redirect(url_for("login"))
    teachers = get_df("data/teachers.csv").to_dict(orient="records")
    students = get_df("data/students.csv").to_dict(orient="records")
    for s in students:
        risk_data = calculate_multi_factor_risk(load_student_profile(s["usn"]))
        s["risk_level"] = risk_data["level"]
        s["dropout_prob"] = risk_data["dropout_prob"]
    parents  = get_df("data/parents.csv").to_dict(orient="records")
    return render_template("admin_dashboard.html",
        admin_name=session.get("name", "Admin"),
        teachers=teachers, students=students, parents=parents)

# ── Teachers ─────────────────────────────────────────────────
@app.route("/admin/teachers/add", methods=["POST"])
def admin_add_teacher():
    if not admin_required(): return redirect(url_for("login"))
    from werkzeug.security import generate_password_hash
    tid  = request.form.get("teacher_id").strip()
    name = request.form.get("name").strip()
    dept = request.form.get("department").strip()
    df = get_df("data/teachers.csv")
    if not df.empty and tid in df["teacher_id"].values:
        flash(f"Teacher ID '{tid}' already exists.", "error")
        return redirect(url_for("admin_dashboard") + "#teachers")
    new_row = {"teacher_id": tid, "name": name, "department": dept,
               "password": generate_password_hash("welcome123")}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_df(df, "data/teachers.csv")
    flash(f"Teacher '{name}' added successfully.")
    return redirect(url_for("admin_dashboard") + "#teachers")

@app.route("/admin/teachers/edit/<teacher_id>", methods=["POST"])
def admin_edit_teacher(teacher_id):
    if not admin_required(): return redirect(url_for("login"))
    from werkzeug.security import generate_password_hash
    df = get_df("data/teachers.csv")
    idx = df[df["teacher_id"] == teacher_id].index
    if idx.empty:
        flash("Teacher not found.", "error")
        return redirect(url_for("admin_dashboard") + "#teachers")
    df.loc[idx[0], "name"]       = request.form.get("name").strip()
    df.loc[idx[0], "department"] = request.form.get("department").strip()
    new_pwd = request.form.get("password")
    if new_pwd and new_pwd.strip():
        df.loc[idx[0], "password"] = generate_password_hash(new_pwd)
    save_df(df, "data/teachers.csv")
    flash(f"Teacher '{teacher_id}' updated successfully.")
    return redirect(url_for("admin_dashboard") + "#teachers")

@app.route("/admin/teachers/delete/<teacher_id>", methods=["POST"])
def admin_delete_teacher(teacher_id):
    if not admin_required(): return redirect(url_for("login"))
    df = get_df("data/teachers.csv")
    df = df[df["teacher_id"] != teacher_id]
    save_df(df, "data/teachers.csv")
    flash(f"Teacher '{teacher_id}' removed.")
    return redirect(url_for("admin_dashboard") + "#teachers")

# ── Students ──────────────────────────────────────────────────
@app.route("/admin/students/add", methods=["POST"])
def admin_add_student():
    if not admin_required(): return redirect(url_for("login"))
    from werkzeug.security import generate_password_hash
    usn    = request.form.get("usn").strip()
    name   = request.form.get("name").strip()
    branch = request.form.get("branch").strip()
    sem    = request.form.get("semester")
    mentor = request.form.get("mentor_id", "")
    df = get_df("data/students.csv")
    if not df.empty and usn in df["usn"].values:
        flash(f"USN '{usn}' already exists.", "error")
        return redirect(url_for("admin_dashboard") + "#students")
    new_row = {"usn": usn, "name": name, "branch": branch, "semester": sem,
               "mentor_id": mentor, "password": generate_password_hash("welcome123")}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_df(df, "data/students.csv")
    flash(f"Student '{name}' ({usn}) added successfully.")
    return redirect(url_for("admin_dashboard") + "#students")

@app.route("/admin/students/edit/<usn>", methods=["POST"])
def admin_edit_student(usn):
    if not admin_required(): return redirect(url_for("login"))
    from werkzeug.security import generate_password_hash
    df = get_df("data/students.csv")
    idx = df[df["usn"] == usn].index
    if idx.empty:
        flash("Student not found.", "error")
        return redirect(url_for("admin_dashboard") + "#students")
    df.loc[idx[0], "name"]      = request.form.get("name").strip()
    df.loc[idx[0], "branch"]    = request.form.get("branch").strip()
    df.loc[idx[0], "semester"]  = request.form.get("semester")
    df.loc[idx[0], "mentor_id"] = request.form.get("mentor_id", "")
    new_pwd = request.form.get("password")
    if new_pwd and new_pwd.strip():
        df.loc[idx[0], "password"] = generate_password_hash(new_pwd)
    save_df(df, "data/students.csv")
    flash(f"Student '{usn}' updated successfully.")
    return redirect(url_for("admin_dashboard") + "#students")

@app.route("/admin/students/delete/<usn>", methods=["POST"])
def admin_delete_student(usn):
    if not admin_required(): return redirect(url_for("login"))
    df = get_df("data/students.csv")
    df = df[df["usn"] != usn]
    save_df(df, "data/students.csv")
    flash(f"Student '{usn}' removed.")
    return redirect(url_for("admin_dashboard") + "#students")

# ── Parents ───────────────────────────────────────────────────
@app.route("/admin/parents/add", methods=["POST"])
def admin_add_parent():
    if not admin_required(): return redirect(url_for("login"))
    from werkzeug.security import generate_password_hash
    pid  = request.form.get("parent_id").strip()
    name = request.form.get("name").strip()
    usn  = request.form.get("usn").strip()
    df = get_df("data/parents.csv")
    if not df.empty and pid in df["parent_id"].values:
        flash(f"Parent ID '{pid}' already exists.", "error")
        return redirect(url_for("admin_dashboard") + "#parents")
    new_row = {"parent_id": pid, "name": name, "usn": usn,
               "password": generate_password_hash("welcome123")}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_df(df, "data/parents.csv")
    flash(f"Parent '{name}' linked to {usn} successfully.")
    return redirect(url_for("admin_dashboard") + "#parents")

@app.route("/admin/parents/delete/<parent_id>", methods=["POST"])
def admin_delete_parent(parent_id):
    if not admin_required(): return redirect(url_for("login"))
    df = get_df("data/parents.csv")
    df = df[df["parent_id"] != parent_id]
    save_df(df, "data/parents.csv")
    flash(f"Parent '{parent_id}' removed.")
    return redirect(url_for("admin_dashboard") + "#parents")

# ── Clear / Wipe Routes ───────────────────────────────────────
CLEAR_TARGETS = {
    "weekly_feedback":  ("data/weekly_feedback.csv",   ["usn","week","stress","academic_issues","personal_issues","understanding_level","missed_classes","need_help"]),
    "academic_marks":   ("data/academic_marks.csv",    ["usn","semester","subject_name","cie","see"]),
    "academic_summary": ("data/academic_summary.csv",  ["usn","semester","attendance","sgpa","cgpa","activity_points"]),
    "meeting_requests": ("data/meeting_requests.csv",  ["parent_id","teacher_id","usn","status"]),
    "notifications":    ("data/notifications.csv",     ["teacher_id","active"]),
    "students":         ("data/students.csv",           ["usn","name","branch","semester","mentor_id","password"]),
    "teachers":         ("data/teachers.csv",           ["teacher_id","name","department","password"]),
    "parents":          ("data/parents.csv",            ["parent_id","name","usn","password"]),
}

@app.route("/admin/clear/<target>", methods=["POST"])
def admin_clear(target):
    if not admin_required(): return redirect(url_for("login"))
    if target not in CLEAR_TARGETS:
        flash("Unknown clear target.", "error")
        return redirect(url_for("admin_dashboard") + "#danger")
    path, cols = CLEAR_TARGETS[target]
    save_df(pd.DataFrame(columns=cols), path)
    flash(f"'{target.replace('_', ' ').title()}' data cleared successfully.")
    return redirect(url_for("admin_dashboard") + "#danger")

@app.route("/admin/clear/everything", methods=["POST"])
def admin_clear_everything():
    if not admin_required(): return redirect(url_for("login"))
    for path, cols in CLEAR_TARGETS.values():
        save_df(pd.DataFrame(columns=cols), path)
    flash("All data wiped. Admin account preserved.", "error")
    return redirect(url_for("admin_dashboard"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)