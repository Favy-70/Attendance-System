import streamlit as st
import csv
import cv2, pickle, os
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime, time


FOLDER = r"C:\Users\favour\attendance_record"
os.makedirs(FOLDER, exist_ok=True)

def get_attendance_file():
    return os.path.join(FOLDER, f'attendance_{datetime.now().date()}.csv')

def load_marked(file):
    if os.path.exists(file):
        return set(pd.read_csv(file)['Name'])
    return set()

current_file = get_attendance_file()
already_marked = load_marked(current_file)

CUTOFF = time(8,0)
def get_status():
    return "Present" if datetime.now().time() <= CUTOFF else "Late"

def mark_attendance(name):
    global current_file, already_marked
    if name == 'Unknown':
        return
    file = get_attendance_file()
    if file != current_file:
        current_file = file
        already_marked = load_marked(file)
    if name in already_marked:
        return
    already_marked.add(name)
    now = datetime.now()
    rec = {'Name': name, 'Date': now.strftime('%Y-%m-%d'),
           'Time': now.strftime('%H:%M:%S'), 'Status': get_status()}
    pd.DataFrame([rec]).to_csv(
        file, mode='a',
        header=not os.path.exists(file), index=False)

st.title('Face Recognition Attendance System')
st.subheader('Traitz Acedemy | Hort 7.0')


with open(r'C:\Users\favour\Desktop\Internship 2\encoding.pkl', 'rb') as f:
    known_encodings, known_names = pickle.load(f)

def mark_absentees():
    today = datetime.now().strftime("%Y-%m-%d")
    path = get_attendance_file()

    present = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            present = {row["Name"] for row in csv.DictReader(f)}

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if os.path.getsize(path) == 0:
            writer.writerow(["Name", "Date", "Time", "Status"])
        for name in known_names:
            if name not in present:
                writer.writerow([name, today, "--:--:--", "Absent"])

if st.button("Mark Absentees"):
    mark_absentees()
    st.success("Absentees Marked")

FRAME_WINDOW = st.image([])
run = st.checkbox('Start Camera')

if run:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb_small)
        encs = face_recognition.face_encodings(rgb_small, locs)
        for enc, loc in zip(encs, locs):
            matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
            dists = face_recognition.face_distance(known_encodings, enc)
            best = np.argmin(dists)
            name = known_names[best] if matches[best] else 'Unknown'
            mark_attendance(name)
            top, right, bottom, left = [v * 4 for v in loc]
            color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
