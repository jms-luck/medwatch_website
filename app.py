from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import cv2
import time
from pathlib import Path
import os
from twilio.rest import Client
import pymssql
import hashlib

def send_whatsapp_message(timestamp: str):
    # Prefer environment variables to avoid committing secrets
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    from_number = os.getenv('TWILIO_WHATSAPP_FROM', 'whatsapp:+')
    to_number = os.getenv('TWILIO_WHATSAPP_TO', 'whatsapp:+')

    client = Client(account_sid, auth_token)

    # Send a simple body message to avoid content variable errors
    message = client.messages.create(
        from_=from_number,
        to=to_number,
        body='Fall detected'
    )
    print(message.sid)


from utils import (
    initialize_preprocessing,
    initialize_video,
    is_night,
    process_frame
)
from detection import detect_persons_yolo
from posture import *

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for live stats
current_stats = {
    "status": "Detecting...",
    "angle": 0,
    "velocity": 0
}

HTML_DIR = (Path(__file__).parent / "frontend" / "src").resolve()


def render_html(template_name: str) -> str:
    """Return HTML file contents with template adjustments."""
    template_path = (HTML_DIR / template_name).resolve()
    if HTML_DIR not in template_path.parents:
        raise HTTPException(status_code=404, detail="Page not found")
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Page not found")

    html_content = template_path.read_text(encoding="utf-8")
    return html_content.replace("{{ url_for('video_feed') }}", "/video_feed")

def generate_frames():
    global current_stats
    initialize_preprocessing()
    mp_drawing, mp_pose, pose = initialize_mediapipe()

    cap, fps, w, h = initialize_video(
        # r"C:\Users\meena\Desktop\dock\downloads\test.mp4"
        r"C:\Users\meena\Desktop\dock\downloads\r.mp4"
    )

    last_center = None
    last_timestamp = None
    last_velocity = 0.0

    lying_start_time = None
    fall_detected_time = None
    last_alert_time = 0

    VELOCITY_THRESHOLD = 20      # px/sec - lowered to match actual fall velocities
    LYING_ANGLE_THRESHOLD = 60
    LYING_TIME_THRESHOLD = 5      # seconds - reduced for faster alert
    ALERT_COOLDOWN = 20          # seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        night = is_night(frame)
        processed_frame = process_frame(frame, 0, night)
        ph, pw, _ = processed_frame.shape

        boxes = detect_persons_yolo(processed_frame)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        rgb_proc = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_proc)

        status = "No Person"
        angle_value = 0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            shoulder = get_xy(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], pw, ph
            )
            hip = get_xy(
                landmarks[mp_pose.PoseLandmark.LEFT_HIP], pw, ph
            )

            torso_angle = torso_tilt_angle(shoulder, hip)
            angle_value = torso_angle

            now = time.time()

            # ---------------- VELOCITY ----------------
            if last_center and last_timestamp:
                dt = max(now - last_timestamp, 1e-3)
                dx = hip[0] - last_center[0]
                dy = hip[1] - last_center[1]
                dist = (dx * dx + dy * dy) ** 0.5
                last_velocity = dist / dt

            last_center = hip
            last_timestamp = now

            # ---------------- POSTURE LOGIC ----------------
            if torso_angle < 20:
                status = "STANDING"
                lying_start_time = None
                fall_detected_time = None

            elif torso_angle < 60:
                status = "SITTING"
                lying_start_time = None
                fall_detected_time = None

            else:
                status = "LYING / FALL"
                print(f"Velocity: {last_velocity:.1f}, Fall time: {fall_detected_time}")
                # Phase 1: detect sudden fall
                if last_velocity > VELOCITY_THRESHOLD and fall_detected_time is None:
                    fall_detected_time = now
                    print(f"🔥 High velocity detected! {int(last_velocity)} px/s (threshold: {VELOCITY_THRESHOLD})")

                # Phase 2: lying confirmation
                if lying_start_time is None:
                    lying_start_time = now
                    print(f"⏱️ Started lying timer")

                lying_duration = now - lying_start_time
                print(f"📊 Status: {status} | Velocity: {int(last_velocity)} | Lying: {lying_duration:.1f}s | Fall detected: {fall_detected_time is not None}")

                # 🚨 FINAL ALERT CONDITION
                if (
                    fall_detected_time is not None and
                    lying_duration >= LYING_TIME_THRESHOLD and
                    now - last_alert_time > ALERT_COOLDOWN
                ):
                    print(f"🚨 SENDING WHATSAPP ALERT!")
                    try:
                        send_whatsapp_message(timestamp=time.strftime('%H:%M:%S'))
                        last_alert_time = now
                        fall_detected_time = None   # reset after alert
                        print("✅ Alert sent successfully")
                    except Exception as e:
                        print(f"❌ Alert failed: {e}")

            mp_drawing.draw_landmarks(
                processed_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            cv2.line(processed_frame, shoulder, hip, (255, 0, 0), 3)

        # ---------------- HUD ----------------
        # Update global stats for API
        current_stats["status"] = status
        current_stats["angle"] = int(angle_value)
        current_stats["velocity"] = int(last_velocity)

        cv2.putText(processed_frame, f"Angle: {int(angle_value)} deg",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.putText(processed_frame, f"Status: {status}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

        cv2.putText(processed_frame, f"Velocity: {int(last_velocity)} px/s",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        if lying_start_time:
            cv2.putText(processed_frame,
                        f"Lying Time: {int(time.time()-lying_start_time)} s",
                        (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

    cap.release()


@app.get('/', response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "frontend"/ "src" / "login.html"
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return html_content

@app.get('/home', response_class=HTMLResponse)
async def home():
    return render_html("home.html")

@app.get('/stream', response_class=HTMLResponse)
async def stream():
    return render_html("home.html")


@app.get('/video_feed')
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get('/stats')
async def stats():
    return current_stats


@app.get('/{page_name}.html', response_class=HTMLResponse)
async def serve_html(page_name: str):
    return render_html(f"{page_name}.html")

class FamilyData(BaseModel):
    family_id: str
    group_name: str
    role: str
    mail_id: str
    cam_url: str
    phone_number: str
    name: str
    group_pass: str


class UserRegistration(BaseModel):
    username: str
    mail_id: str
    password: str
    name: Optional[str] = None
    age: Optional[int] = None
    phone_number: Optional[str] = None


class UserLogin(BaseModel):
    mail_id: str
    password: str


class FamilyMemberData(BaseModel):
    family_id: str
    name: str
    mail_id: str
    phone_number: str

def get_connection(database="medu"):
    """
    Returns a connection to the Azure SQL database.
    
    Args:
        database (str): Database name to connect to. Defaults to "medu".
    
    Returns:
        Connection object or None if connection fails.
    """
    try:
        conn = pymssql.connect(
            server="adminsss.database.windows.net",
            user="admin123",
            password="Password123",
            database=database,
            port=1433,
            autocommit=True
        )
        print(f"✅ Connected to Azure SQL ({database})")
        return conn
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return None


def hash_password(password: str) -> str:
    """Return a simple SHA-256 hash of the password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


@app.post("/register_family")
def register_family(family: FamilyData):
    conn = get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        cursor = conn.cursor()

        # 1️⃣ Insert into Family master table
        insert_sql = """
        INSERT INTO [Family]
        (FamilyId, group_name, role, mail_id, cam_url, phone_number, name, group_pass)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(
            insert_sql,
            (
                family.family_id,
                family.group_name,
                family.role,
                family.mail_id,
                family.cam_url,
                family.phone_number,
                family.name,
                family.group_pass,
            )
        )

        # 2️⃣ Create family-specific members table
        table_name = f"Family_{family.family_id}"

        create_table_sql = f"""
        IF NOT EXISTS (
            SELECT * FROM sysobjects WHERE name='{table_name}' AND xtype='U'
        )
        CREATE TABLE [{table_name}] (
            member_id INT IDENTITY(1,1) PRIMARY KEY,
            name NVARCHAR(100),
            mail_id NVARCHAR(100),
            phone_number NVARCHAR(20),
            created_at DATETIME DEFAULT GETDATE()
        )
        """

        cursor.execute(create_table_sql)

        conn.commit()
        return {"message": "Family registered and family table created successfully"}

    except Exception as exc:
        conn.rollback()
        import traceback
        error_details = f"{str(exc)}\n{traceback.format_exc()}"
        print(f"Error in register_family: {error_details}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(exc)}")
    finally:
        conn.close()


@app.get("/families")
def list_families():
    conn = get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        with conn.cursor(as_dict=True) as cursor:
            cursor.execute(
                "SELECT FamilyId as family_id, group_name, role, mail_id, cam_url, phone_number, name, group_pass FROM [Family] ORDER BY group_name"
            )
            rows = cursor.fetchall()
            return {"families": rows or []}
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to fetch families") from exc
    finally:
        conn.close()

@app.post("/join_family")
def join_family(family: FamilyData):
    conn = get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        with conn.cursor(as_dict=True) as cursor:
            cursor.execute(
                "SELECT FamilyId as family_id FROM [Family] WHERE group_name = %s AND group_pass = %s",
                (family.group_name, family.group_pass),
            )
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Family not found or invalid password")
            return {"message": "Joined family successfully", "family_id": row["family_id"]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to join family") from exc
    finally:
        conn.close()


@app.post("/register")
def register_user(user: UserRegistration):
    conn = get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        with conn.cursor() as cursor:
            insert_sql = (
                "INSERT INTO [Users] (username, mail_id, password, name, age, phone_number) "
                "VALUES (%s, %s, %s, %s, %s, %s)"
            )
            cursor.execute(
                insert_sql,
                (
                    user.username,
                    user.mail_id,
                    hash_password(user.password),
                    user.name,
                    user.age,
                    user.phone_number,
                ),
            )
        return {"message": "User registered successfully"}
    except pymssql.IntegrityError as exc:  # likely duplicate username
        raise HTTPException(status_code=400, detail="Username already exists") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to register user") from exc
    finally:
        conn.close()


@app.post("/login")
def login_user(user: UserLogin):
    conn = get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        with conn.cursor(as_dict=True) as cursor:
            cursor.execute(
                "SELECT username, mail_id, name, age, phone_number FROM [Users] WHERE mail_id = %s AND password = %s",
                (user.mail_id, hash_password(user.password)),
            )
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=401, detail="Invalid credentials")
            return {"message": "Login successful", "user": row}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to login") from exc
    finally:
        conn.close()


@app.post("/add_family_member")
def add_family_member(member: FamilyMemberData):
    conn = get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        table_name = f"Family_{member.family_id}"
        
        with conn.cursor() as cursor:
            # Check if family exists in Family table first
            cursor.execute(
                "SELECT FamilyId FROM [Family] WHERE FamilyId = %s",
                (member.family_id,)
            )
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail=f"Family with ID {member.family_id} does not exist")
            
            # Create table if it doesn't exist
            create_table_sql = f"""
            IF NOT EXISTS (
                SELECT * FROM sysobjects WHERE name='{table_name}' AND xtype='U'
            )
            CREATE TABLE [{table_name}] (
                member_id INT IDENTITY(1,1) PRIMARY KEY,
                name NVARCHAR(100),
                mail_id NVARCHAR(100),
                phone_number NVARCHAR(20),
                created_at DATETIME DEFAULT GETDATE()
            )
            """
            cursor.execute(create_table_sql)
            
            # Insert member into family table
            insert_sql = f"""
            INSERT INTO [{table_name}] (name, mail_id, phone_number)
            VALUES (%s, %s, %s)
            """
            cursor.execute(
                insert_sql,
                (member.name, member.mail_id, member.phone_number)
            )
            conn.commit()
            
        return {"message": f"Successfully added member to family {member.family_id}"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to add family member: {str(exc)}") from exc
    finally:
        conn.close()

# Mount static files at the end to prevent route interference
app.mount("/css", StaticFiles(directory=Path(__file__).parent / "frontend" / "src" / "css"), name="css")
app.mount("/js", StaticFiles(directory=Path(__file__).parent / "frontend" / "src" / "js"), name="js")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
