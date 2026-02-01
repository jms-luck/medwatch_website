import hashlib
from typing import Optional

import pymssql
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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


if __name__ == "__main__":
    conn = get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    uvicorn.run(app, host="0.0.0.0", port=8000)