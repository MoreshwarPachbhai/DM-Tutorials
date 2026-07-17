import random
import pandas as pd

rows = []

for _ in range(1000):

    attendance = random.randint(40,100)
    grades = random.randint(35,100)
    study_hours = random.randint(1,8)
    backlogs = random.randint(0,6)
    income = random.choice(["Low","Medium","High"])
    scholarship = random.choice(["Yes","No"])
    internet = random.choice(["Yes","No"])
    gender = random.choice(["Male","Female"])

    dropout = 0

    if attendance < 60:
        dropout = 1

    if grades < 50:
        dropout = 1

    if backlogs >= 3:
        dropout = 1

    if study_hours <= 2:
        dropout = 1

    rows.append([
        attendance,
        grades,
        study_hours,
        backlogs,
        income,
        scholarship,
        internet,
        gender,
        dropout
    ])

columns = [
    "Attendance",
    "Grades",
    "StudyHours",
    "Backlogs",
    "FamilyIncome",
    "Scholarship",
    "Internet",
    "Gender",
    "Dropout"
]

df = pd.DataFrame(rows, columns=columns)

df.to_csv("student_dropout.csv", index=False)

print("Dataset Generated Successfully")