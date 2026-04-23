#include <Wire.h>

const uint8_t MPU_ADDR = 0x68;

// --- FINAL PERFECTED OFFSETS ---
const float AX_OFF = 1060.08;
const float AY_OFF = 23.86;
const float AZ_OFF = 353.86;
const float GX_OFF = 282.60;
const float GY_OFF = 473.91;
const float GZ_OFF = 4.78;

// 16:38:52.494 -> === Calibration Offsets ===
// 16:38:52.494 -> Accel  X: 1060.08  Y: 23.86  Z: 353.86
// 16:38:52.494 -> Gyro   X: 282.60  Y: 473.91  Z: 4.78

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission();
}

void loop() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MPU_ADDR, (size_t)14, (bool)true);

  int16_t rax = Wire.read() << 8 | Wire.read();
  int16_t ray = Wire.read() << 8 | Wire.read();
  int16_t raz = Wire.read() << 8 | Wire.read();
  Wire.read(); Wire.read(); 
  int16_t rgx = Wire.read() << 8 | Wire.read();
  int16_t rgy = Wire.read() << 8 | Wire.read();
  int16_t rgz = Wire.read() << 8 | Wire.read();

  // Convert to real units
  float ax = (rax - AX_OFF) / 16384.0; 
  float ay = (ray - AY_OFF) / 16384.0;
  float az = (raz - AZ_OFF) / 16384.0;

  float gx = (rgx - GX_OFF) / 131.0;
  float gy = (rgy - GY_OFF) / 131.0;
  float gz = (rgz - GZ_OFF) / 131.0;

  // --- APPLY DEADZONE (Filtering noise) ---
  if (abs(gx) < 0.05) gx = 0;
  if (abs(gy) < 0.05) gy = 0;
  if (abs(gz) < 0.05) gz = 0;

  // Displaying X, Y, Z for both sets
  Serial.print("ACCEL > X: "); Serial.print(ax, 3);
  Serial.print(" | Y: ");      Serial.print(ay, 3);
  Serial.print(" | Z: ");      Serial.print(az, 3);
  
  Serial.print("  ||  GYRO > X: ");  Serial.print(gx, 2);
  Serial.print(" | Y: ");           Serial.print(gy, 2);
  Serial.print(" | Z: ");           Serial.println(gz, 2);

  delay(100);
}