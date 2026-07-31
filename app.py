import json
import math
import os
import subprocess
import time
import tkinter as tk
import webbrowser
from collections import deque
from tkinter import messagebox, ttk

import cv2
import mediapipe as mp
from PIL import Image, ImageTk


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

GESTURES = {
    "thumbs_up": "엄지손가락 올리기",
    "both_hands_up": "양손 들기",
    "open_hand": "손바닥 펴기",
    "smile": "미소",
    "wink": "윙크",
    "mouth_open": "입 벌리기",
    "brow_raise": "눈썹 올리기",
}

ACTION_TYPES = {
    "youtube": "유튜브 영상 열기",
    "url": "웹사이트 열기",
    "app": "Windows 앱 실행",
    "file": "파일 열기",
    "notification": "앱 안에서 알림 표시",
}

# 표정은 짧게 안정화하고, 손동작은 더 빠르게 반응하도록 행동별 시간을 둡니다.
DEFAULT_STABLE_FRAMES = {
    "smile": 4,
    "wink": 4,
    "mouth_open": 4,
    "brow_raise": 4,
    "thumbs_up": 3,
    "open_hand": 3,
    "both_hands_up": 3,
}

DEFAULT_ACTIONS = [
    {"name": "유튜브 재생", "gesture": "thumbs_up", "type": "youtube", "target": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "priority": 50},
    {"name": "유튜브 홈 열기", "gesture": "both_hands_up", "type": "url", "target": "https://www.youtube.com", "priority": 50},
    {"name": "메모장 열기", "gesture": "open_hand", "type": "app", "target": "notepad.exe", "priority": 50},
    {"name": "계산기 열기", "gesture": "smile", "type": "app", "target": "calc.exe", "priority": 50},
    {"name": "문서 폴더 열기", "gesture": "wink", "type": "file", "target": os.path.expanduser("~/Documents"), "priority": 50},
    {"name": "집중 모드 알림", "gesture": "brow_raise", "type": "notification", "target": "집중 모드를 시작할까요?", "priority": 50},
]


class GestureDetector:
    """MediaPipe Hands/Pose/Face Mesh를 사용한 실시간 동작 감지기."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.65, min_tracking_confidence=0.6
        )
        self.face = self.mp_face.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        # 평균 프레임 수를 줄여 표정 판정 지연을 낮춥니다.
        self.face_history = deque(maxlen=3)

    @staticmethod
    def dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    @staticmethod
    def ratio(face, vertical_a, vertical_b, horizontal_a, horizontal_b):
        return GestureDetector.dist(face[vertical_a], face[vertical_b]) / max(GestureDetector.dist(face[horizontal_a], face[horizontal_b]), 1e-6)

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = self.hands.process(rgb)
        face_result = self.face.process(rgb)
        pose = self.pose.process(rgb)
        labels = set()

        if hands.multi_hand_landmarks:
            if len(hands.multi_hand_landmarks) >= 2:
                labels.add("two_hands")
            for hand in hands.multi_hand_landmarks:
                lm = hand.landmark
                # 손가락 끝과 관절의 상대 위치를 비교해 손 모양을 판단합니다.
                fingers_up = sum(lm[i].y < lm[i - 2].y for i in (8, 12, 16, 20))
                thumb_up = lm[4].y < lm[3].y and lm[4].y < lm[2].y
                if thumb_up and fingers_up <= 1:
                    labels.add("thumbs_up")
                if fingers_up >= 3:
                    labels.add("open_hand")

        if pose.pose_landmarks:
            p = pose.pose_landmarks.landmark
            if p[15].y < p[11].y and p[16].y < p[12].y:
                labels.add("both_hands_up")

        if face_result.multi_face_landmarks:
            face = face_result.multi_face_landmarks[0].landmark
            # FaceMesh는 468개 기본 랜드마크(+ refine_landmarks 사용 시 홍채 포함)를 제공합니다.
            mouth_width = self.dist(face[61], face[291])
            mouth_open = self.ratio(face, 13, 14, 61, 291)
            face_width = max(self.dist(face[33], face[263]), 1e-6)
            smile = mouth_width / face_width
            # 입꼬리가 입 중앙보다 위로 올라간 정도를 함께 봅니다.
            mouth_center_y = (face[13].y + face[14].y) / 2
            corner_center_y = (face[61].y + face[291].y) / 2
            corner_lift = (mouth_center_y - corner_center_y) / face_width
            left_eye = self.ratio(face, 159, 145, 33, 133)
            right_eye = self.ratio(face, 386, 374, 362, 263)
            left_brow = self.dist(face[105], face[159]) / max(self.dist(face[33], face[263]), 1e-6)
            right_brow = self.dist(face[334], face[386]) / max(self.dist(face[33], face[263]), 1e-6)
            self.face_history.append((smile, corner_lift, mouth_open, left_eye, right_eye, left_brow, right_brow))
            avg = [sum(x[i] for x in self.face_history) / len(self.face_history) for i in range(7)]
            # 입 너비만 큰 중립 표정/말하기를 제외하고, 실제 웃음처럼 입꼬리가 올라간 경우만 인정합니다.
            if avg[0] > 0.40 and avg[1] > 0.012:
                labels.add("smile")
            if avg[2] > 0.12:
                labels.add("mouth_open")
            if avg[3] < 0.18 or avg[4] < 0.18:
                labels.add("wink")
            if avg[5] > 0.22 or avg[6] > 0.22:
                labels.add("brow_raise")

        return labels, hands, face_result, pose

    def close(self):
        self.hands.close()
        self.face.close()
        self.pose.close()


class ActionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Link - 행동 연동 앱")
        self.root.geometry("1280x820")
        self.root.minsize(1050, 700)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.cap = None
        self.detector = GestureDetector()
        self.running = False
        self.last_trigger = {}
        self.pending_prompt = False
        self.stable_frame_count = 0
        self.release_frame_count = 0
        self.last_candidate_gesture = None
        self.gesture_armed = True
        self.config = self.load_config()
        self.selected_index = None
        self.build_ui()

    def load_config(self):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("cooldown_seconds", 5)
            data.setdefault("stable_frames", 4)
            data.setdefault("release_frames", 5)
            data.setdefault("actions", [])
            return data
        except (OSError, json.JSONDecodeError):
            return {"cooldown_seconds": 5, "stable_frames": 4, "release_frames": 5, "actions": [dict(x) for x in DEFAULT_ACTIONS]}

    def build_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        self.run_tab = ttk.Frame(self.notebook, padding=10)
        self.edit_tab = ttk.Frame(self.notebook, padding=16)
        self.notebook.add(self.run_tab, text="실행 화면")
        self.notebook.add(self.edit_tab, text="행동 연결 편집")
        self.build_run_tab()
        self.build_edit_tab()

    def build_run_tab(self):
        self.video = tk.Label(self.run_tab, bg="#111111")
        self.video.pack(side="left", fill="both", expand=True, padx=(0, 12))
        panel = ttk.Frame(self.run_tab, width=300)
        panel.pack(side="right", fill="y")
        ttk.Label(panel, text="Gesture Link", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(panel, text="행동을 인식하면 연결 작업 실행 전에 확인합니다.", wraplength=290).pack(anchor="w", pady=(4, 18))
        self.status = tk.StringVar(value="카메라를 시작하세요.")
        ttk.Label(panel, textvariable=self.status, foreground="#155e75", wraplength=290).pack(anchor="w", pady=6)
        ttk.Button(panel, text="카메라 시작", command=self.start).pack(fill="x", pady=4)
        ttk.Button(panel, text="카메라 중지", command=self.stop).pack(fill="x", pady=4)
        ttk.Button(panel, text="연결 편집 열기", command=lambda: self.notebook.select(self.edit_tab)).pack(fill="x", pady=(4, 16))
        ttk.Label(panel, text="현재 연결", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.action_text = tk.Text(panel, width=36, height=18, state="disabled", wrap="word")
        self.action_text.pack(fill="both", expand=True, pady=8)
        self.refresh_action_text()

    def build_edit_tab(self):
        left = ttk.Frame(self.edit_tab)
        left.pack(side="left", fill="both", expand=True, padx=(0, 18))
        right = ttk.LabelFrame(self.edit_tab, text="연결 설정", padding=14)
        right.pack(side="right", fill="y")
        ttk.Label(left, text="행동 → 작업 연결", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(left, text="왼쪽에서 연결을 선택하거나 새 연결을 만든 뒤, 오른쪽에서 직관적으로 설정하세요.").pack(anchor="w", pady=(4, 12))
        columns = ("name", "gesture", "type", "priority", "target")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", height=20)
        for col, title, width in (("name", "연결 이름", 180), ("gesture", "감지 행동", 140), ("type", "작업 유형", 160), ("priority", "우선순위", 80), ("target", "대상", 330)):
            self.tree.heading(col, text=title)
            self.tree.column(col, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.select_action)
        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=10)
        ttk.Button(buttons, text="새 연결", command=self.new_action).pack(side="left")
        ttk.Button(buttons, text="선택 삭제", command=self.delete_action).pack(side="left", padx=6)
        ttk.Button(buttons, text="기본 프리셋 추가", command=self.add_presets).pack(side="left")
        ttk.Button(buttons, text="설정 저장", command=self.save_config).pack(side="right")

        self.name_var = tk.StringVar()
        self.gesture_var = tk.StringVar()
        self.type_var = tk.StringVar()
        self.priority_var = tk.StringVar(value="50")
        self.target_var = tk.StringVar()
        self.description_var = tk.StringVar(value="")
        self.form_row(right, "연결 이름", ttk.Entry(right, textvariable=self.name_var, width=34), 0)
        gesture_box = ttk.Combobox(right, textvariable=self.gesture_var, values=list(GESTURES), state="readonly", width=31)
        self.form_row(right, "감지 행동", gesture_box, 1)
        ttk.Label(right, text="= " + GESTURES.get(self.gesture_var.get(), "행동을 선택하세요")).grid(row=2, column=1, sticky="w", pady=(0, 8))
        type_box = ttk.Combobox(right, textvariable=self.type_var, values=list(ACTION_TYPES), state="readonly", width=31)
        self.form_row(right, "실행 작업", type_box, 3)
        self.form_row(right, "우선순위", ttk.Entry(right, textvariable=self.priority_var, width=34), 4)
        ttk.Label(right, text="숫자가 높을수록 동시에 감지될 때 먼저 선택됩니다.", foreground="#666666", wraplength=250).grid(row=5, column=1, sticky="w")
        ttk.Label(right, text="대상", width=12).grid(row=6, column=0, sticky="nw", pady=8)
        ttk.Entry(right, textvariable=self.target_var, width=34).grid(row=6, column=1, sticky="ew", pady=8)
        ttk.Label(right, text="URL, 프로그램명, 파일 경로 또는 알림 문구", foreground="#666666", wraplength=250).grid(row=7, column=1, sticky="w")
        ttk.Label(right, text="설명", width=12).grid(row=8, column=0, sticky="nw", pady=8)
        ttk.Entry(right, textvariable=self.description_var, width=34).grid(row=8, column=1, sticky="ew", pady=8)
        ttk.Button(right, text="이 연결 적용", command=self.apply_form).grid(row=9, column=1, sticky="e", pady=(18, 0))
        right.columnconfigure(1, weight=1)
        self.gesture_var.trace_add("write", lambda *_: self.refresh_gesture_hint(right))
        self.populate_tree()

    def form_row(self, parent, label, widget, row):
        ttk.Label(parent, text=label, width=12).grid(row=row, column=0, sticky="w", pady=8)
        widget.grid(row=row, column=1, sticky="ew", pady=8)

    def refresh_gesture_hint(self, parent):
        # 편집 폼의 행동 설명을 갱신합니다.
        for child in parent.grid_slaves(row=2, column=1):
            child.configure(text="= " + GESTURES.get(self.gesture_var.get(), "행동을 선택하세요"))

    def populate_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, action in enumerate(self.config.get("actions", [])):
            self.tree.insert("", "end", iid=str(i), values=(action.get("name", ""), GESTURES.get(action.get("gesture"), action.get("gesture", "")), ACTION_TYPES.get(action.get("type"), action.get("type", "")), action.get("priority", 50), action.get("target", "")))

    def select_action(self, _event=None):
        selected = self.tree.selection()
        if not selected:
            return
        self.selected_index = int(selected[0])
        action = self.config["actions"][self.selected_index]
        self.name_var.set(action.get("name", ""))
        self.gesture_var.set(action.get("gesture", ""))
        self.type_var.set(action.get("type", ""))
        self.priority_var.set(str(action.get("priority", 50)))
        self.target_var.set(action.get("target", ""))
        self.description_var.set(action.get("description", ""))

    def new_action(self):
        self.selected_index = None
        self.tree.selection_remove(self.tree.selection())
        self.name_var.set("새 연결")
        self.gesture_var.set("smile")
        self.type_var.set("notification")
        self.priority_var.set("50")
        self.target_var.set("새 행동이 감지되었습니다.")
        self.description_var.set("")

    def apply_form(self):
        if not self.name_var.get().strip() or not self.gesture_var.get() or not self.type_var.get() or not self.target_var.get().strip():
            messagebox.showwarning("입력 확인", "연결 이름, 감지 행동, 실행 작업, 대상을 모두 입력하세요.")
            return
        try:
            priority = int(self.priority_var.get())
        except ValueError:
            messagebox.showwarning("입력 확인", "우선순위는 정수로 입력하세요.")
            return
        item = {"name": self.name_var.get().strip(), "gesture": self.gesture_var.get(), "type": self.type_var.get(), "target": self.target_var.get().strip(), "priority": priority, "description": self.description_var.get().strip()}
        if self.selected_index is None:
            self.config["actions"].append(item)
            self.selected_index = len(self.config["actions"]) - 1
        else:
            self.config["actions"][self.selected_index] = item
        self.populate_tree()
        self.tree.selection_set(str(self.selected_index))
        self.refresh_action_text()
        self.status.set("연결을 적용했습니다. 저장 버튼을 눌러 파일에도 저장하세요.")

    def delete_action(self):
        if self.selected_index is None:
            return
        if messagebox.askyesno("연결 삭제", "선택한 연결을 삭제할까요?"):
            self.config["actions"].pop(self.selected_index)
            self.selected_index = None
            self.populate_tree()
            self.refresh_action_text()

    def add_presets(self):
        existing = {a.get("name") for a in self.config["actions"]}
        added = 0
        for preset in DEFAULT_ACTIONS:
            if preset["name"] not in existing:
                self.config["actions"].append(dict(preset))
                added += 1
        self.populate_tree()
        self.refresh_action_text()
        messagebox.showinfo("기본 프리셋", f"{added}개의 기본 연결을 추가했습니다. 설정 저장을 눌러 보관하세요.")

    def save_config(self):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        self.refresh_action_text()
        self.status.set("연결 설정을 저장했습니다.")
        messagebox.showinfo("저장 완료", "연결 설정을 config.json에 저장했습니다.")

    def refresh_action_text(self):
        if not hasattr(self, "action_text"):
            return
        self.action_text.configure(state="normal")
        self.action_text.delete("1.0", "end")
        for action in self.config.get("actions", []):
            self.action_text.insert("end", f"• {action.get('name')}\n  {GESTURES.get(action.get('gesture'), action.get('gesture'))} → {ACTION_TYPES.get(action.get('type'), action.get('type'))}\n  {action.get('target')}\n\n")
        self.action_text.configure(state="disabled")

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("카메라 오류", "카메라를 열 수 없습니다.")
            return
        self.running = True
        self.status.set("인식 중... 행동을 취해 보세요.")
        self.update_frame()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if hasattr(self, "video"):
            self.video.configure(image="")
        if hasattr(self, "status"):
            self.status.set("카메라가 중지되었습니다.")

    def update_frame(self):
        if not self.running or not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(100, self.update_frame)
            return
        frame = cv2.flip(frame, 1)
        labels, hands, face, pose = self.detector.detect(frame)
        draw = mp.solutions.drawing_utils
        if hands.multi_hand_landmarks:
            for h in hands.multi_hand_landmarks:
                draw.draw_landmarks(frame, h, self.detector.mp_hands.HAND_CONNECTIONS)
        if face.multi_face_landmarks:
            for f in face.multi_face_landmarks:
                draw.draw_landmarks(frame, f, self.detector.mp_face.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=draw.DrawingSpec(color=(70, 70, 70), thickness=1))
        if pose.pose_landmarks:
            draw.draw_landmarks(frame, pose.pose_landmarks, self.detector.mp_pose.POSE_CONNECTIONS)
        visible = ", ".join(GESTURES.get(label, label) for label in sorted(labels)) if labels else "감지된 행동 없음"
        cv2.putText(frame, visible, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 220, 120), 2)
        image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.video.configure(image=image)
        self.video.image = image
        self.check_actions(labels)
        self.root.after(30, self.update_frame)

    def check_actions(self, labels):
        now = time.time()
        cooldown = float(self.config.get("cooldown_seconds", 5))
        matching = [a for a in self.config.get("actions", []) if a.get("gesture") in labels]
        if not matching:
            self.release_frame_count += 1
            if self.release_frame_count >= int(self.config.get("release_frames", 8)):
                self.gesture_armed = True
                self.stable_frame_count = 0
                self.last_candidate_gesture = None
            return

        self.release_frame_count = 0
        # 동시에 여러 행동이 감지되면 사용자가 지정한 우선순위가 가장 높은 하나만 선택합니다.
        def priority(action):
            try:
                return int(action.get("priority", 50))
            except (TypeError, ValueError):
                return 50
        matching.sort(key=lambda a: (-priority(a), a.get("gesture", ""), a.get("name", "")))
        candidate = matching[0]
        candidate_gesture = candidate.get("gesture")
        if candidate_gesture != self.last_candidate_gesture:
            self.last_candidate_gesture = candidate_gesture
            self.stable_frame_count = 1
            return
        self.stable_frame_count += 1

        # 같은 표정/동작을 유지하는 동안에는 해제될 때까지 다시 요청하지 않습니다.
        if self.pending_prompt or not self.gesture_armed:
            return
        stable_frames = int(self.config.get("stable_frames", 4))
        stable_frames = int(self.config.get("stable_frames_by_gesture", {}).get(candidate_gesture, DEFAULT_STABLE_FRAMES.get(candidate_gesture, stable_frames)))
        if self.stable_frame_count < stable_frames:
            return
        key = candidate.get("name", candidate_gesture)
        if now - self.last_trigger.get(key, 0) < cooldown:
            return
        self.last_trigger[key] = now
        self.pending_prompt = True
        self.gesture_armed = False
        self.running = False
        self.root.after(0, lambda item=dict(candidate): self.ask_and_execute(item))

    def ask_and_execute(self, item):
        accepted = messagebox.askyesno("연결 작업 확인", f"'{item.get('name')}' 행동을 인식했습니다.\n\n{ACTION_TYPES.get(item.get('type'), item.get('type'))}: {item.get('target')}\n\n실행할까요?")
        if accepted:
            try:
                self.execute(item)
                self.status.set(f"실행 완료: {item.get('name')}")
            except Exception as exc:
                messagebox.showerror("작업 실행 오류", str(exc))
        else:
            self.status.set("작업 실행을 취소했습니다.")
        self.pending_prompt = False
        self.running = bool(self.cap and self.cap.isOpened())
        if self.running:
            self.update_frame()

    @staticmethod
    def execute(item):
        kind, target = item.get("type"), item.get("target", "")
        if kind in ("youtube", "url"):
            webbrowser.open(target)
        elif kind in ("app", "file"):
            subprocess.Popen(target, shell=True)
        elif kind == "notification":
            messagebox.showinfo("Gesture Link 알림", target)
        else:
            raise ValueError(f"지원하지 않는 작업 유형입니다: {kind}")

    def close(self):
        self.stop()
        self.detector.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    ActionApp(root)
    root.mainloop()
