import gradio as gr
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("cleaned_data.csv")
X = df[["study_hours", "attendance"]]
y = df["prev_gpa"]

model = DecisionTreeRegressor(
    max_depth=3, random_state=42)
model.fit(X, y)

def predict(hours, attendance):
    new = pd.DataFrame([[hours, attendance]],
                       columns=X.columns)
    return round(model.predict(new)[0], 2)

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(0, 10, step=1, label="ساعات المذاكرة"),
        gr.Slider(40, 100, step=1, label="نسبة الحضور"),
    ],
    outputs=gr.Number(label="المعدل المتوقع (تقدير تقريبي)"),
    title="مُساعِد: توقّع معدل الطالب",
)
demo.launch()
