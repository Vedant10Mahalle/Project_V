document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("student-form");
    if (!form) return;

    form.addEventListener("submit", async function(e) {
        e.preventDefault();

        // UI State: Show Loader, disable button
        const btn = document.getElementById("submit-btn");
        const loader = document.getElementById("loader");
        btn.disabled = true;
        btn.style.opacity = "0.7";
        loader.style.display = "inline-block";

        let formData = new FormData(this);
        let data = {};

        formData.forEach((value, key) => {
            // Treat empty fields as 0
            data[key] = value.trim() === "" ? 0 : Number(value);
        });

        try {
            let res = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            });

            let result = await res.json();

            // Simulate slight network delay for premium feel
            setTimeout(() => {
                form.style.display = "none";
                const overlay = document.getElementById("result-overlay");
                overlay.style.display = "block";

                if (result.success) {
                    document.getElementById("result-text").innerText = result.message;
                } else {
                    document.getElementById("result-text").innerText = "An error occurred: " + result.message;
                    document.getElementById("result-text").style.color = "#fca5a5";
                }
            }, 600);

        } catch (error) {
            console.error("Submission error:", error);
            alert("Failed to submit data. Check the server connection.");
            
            // Revert state
            btn.disabled = false;
            btn.style.opacity = "1";
            loader.style.display = "none";
        }
    });
});