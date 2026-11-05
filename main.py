import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="스마트폰 사용 시간과 수업 집중도 분석",
    page_icon="📱",
    layout="wide"
)

st.title("📱 스마트폰 사용 시간과 수업 집중도 분석")
st.write("학생들의 하루 스마트폰 사용 시간과 수업 집중도 사이의 관계를 알아보는 프로젝트입니다.")

st.info(
    "처음 표시되는 값은 웹앱 작동을 확인하기 위한 예시 데이터입니다. "
    "실제 수행평가에서는 설문조사로 수집한 데이터로 바꿔서 분석하세요."
)

# 예시 데이터: 실제 조사 결과가 아님
sample_data = pd.DataFrame({
    "스마트폰 사용 시간(시간)": [1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7],
    "수업 집중도(1~5점)": [5, 4, 5, 4, 3, 4, 3, 2, 3, 1]
})

st.header("1. 데이터 입력")
st.write("표의 숫자를 직접 수정하거나, 설문조사 결과가 담긴 CSV 파일을 업로드할 수 있습니다.")

uploaded_file = st.file_uploader("설문조사 CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        data = data.rename(columns={
            "사용 시간": "스마트폰 사용 시간(시간)",
            "집중도": "수업 집중도(1~5점)"
        })
    except Exception as error:
        st.error(f"CSV 파일을 읽는 중 문제가 발생했습니다: {error}")
        data = sample_data.copy()
else:
    data = sample_data.copy()

edited_data = st.data_editor(data, num_rows="dynamic", use_container_width=True)

x_name = "스마트폰 사용 시간(시간)"
y_name = "수업 집중도(1~5점)"

if x_name not in edited_data.columns or y_name not in edited_data.columns:
    st.error(f"표에 '{x_name}'와 '{y_name}' 열이 있어야 합니다.")
    st.stop()

clean_data = edited_data[[x_name, y_name]].copy()
clean_data[x_name] = pd.to_numeric(clean_data[x_name], errors="coerce")
clean_data[y_name] = pd.to_numeric(clean_data[y_name], errors="coerce")
clean_data = clean_data.dropna()
clean_data = clean_data[
    (clean_data[x_name] >= 0)
    & (clean_data[x_name] <= 24)
    & (clean_data[y_name] >= 1)
    & (clean_data[y_name] <= 5)
]

st.caption("분석 전 확인: 사용 시간은 0~24시간, 집중도는 1~5점 범위의 데이터만 분석합니다.")

if len(clean_data) < 3:
    st.warning("선형회귀 분석을 하려면 유효한 데이터가 최소 3개 필요합니다.")
    st.stop()

# 독립변수와 종속변수 설정
X = clean_data[[x_name]]
y = clean_data[y_name]

# 선형회귀 모델 학습
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

# 모델 평가
mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)

st.header("2. 데이터 소개")
metric1, metric2, metric3 = st.columns(3)
metric1.metric("전체 데이터 수", f"{len(clean_data)}명")
metric2.metric("평균 스마트폰 사용 시간", f"{clean_data[x_name].mean():.1f}시간")
metric3.metric("평균 수업 집중도", f"{clean_data[y_name].mean():.2f}점")

st.subheader("입력된 데이터")
st.dataframe(clean_data, use_container_width=True)

st.header("3. 선형회귀 분석")
st.write("독립변수(X)는 하루 스마트폰 사용 시간이고, 종속변수(Y)는 수업 집중도 점수입니다.")

slope = float(model.coef_[0])
intercept = float(model.intercept_)
st.write(f"**회귀식:** 수업 집중도 = {intercept:.2f} + ({slope:.2f} × 스마트폰 사용 시간)")

if slope < 0:
    st.write("입력된 데이터에서는 스마트폰 사용 시간이 길수록 집중도 점수가 낮아지는 경향이 나타납니다.")
elif slope > 0:
    st.write("입력된 데이터에서는 스마트폰 사용 시간이 길수록 집중도 점수가 높아지는 경향이 나타납니다.")
else:
    st.write("입력된 데이터에서는 스마트폰 사용 시간에 따른 집중도 변화가 거의 나타나지 않습니다.")

# Plotly 산점도와 회귀선
chart_data = clean_data.copy()
chart_data["회귀선 예측값"] = model.predict(X)

fig = px.scatter(
    chart_data,
    x=x_name,
    y=y_name,
    title="스마트폰 사용 시간과 수업 집중도의 관계",
    labels={
        x_name: "하루 스마트폰 사용 시간(시간)",
        y_name: "수업 집중도(1~5점)"
    },
    range_y=[0.5, 5.5]
)

line_x = np.linspace(clean_data[x_name].min(), clean_data[x_name].max(), 100)
line_y = model.predict(pd.DataFrame({x_name: line_x}))
line_data = pd.DataFrame({
    "하루 스마트폰 사용 시간(시간)": line_x,
    "예상 수업 집중도": line_y
})

fig.add_scatter(
    x=line_data["하루 스마트폰 사용 시간(시간)"],
    y=line_data["예상 수업 집중도"],
    mode="lines",
    name="선형회귀선"
)
st.plotly_chart(fig, use_container_width=True)

st.header("4. 모델 평가")
result1, result2, result3 = st.columns(3)
result1.metric("결정계수 (R²)", f"{r2:.3f}")
result2.metric("평균 절대 오차 (MAE)", f"{mae:.2f}")
result3.metric("평균 제곱근 오차 (RMSE)", f"{rmse:.2f}")

st.write(
    "R²는 모델이 데이터의 변화를 얼마나 설명하는지 나타냅니다. "
    "MAE와 RMSE는 예측값과 실제값의 차이를 나타내며, 값이 작을수록 오차가 작습니다."
)

st.header("5. 새로운 사용 시간으로 집중도 예측하기")
hours = st.number_input(
    "하루 스마트폰 사용 시간(시간)을 입력하세요.",
    min_value=0.0,
    max_value=24.0,
    value=3.0,
    step=0.5
)

if st.button("예상 집중도 확인"):
    predicted_score = float(model.predict(pd.DataFrame({x_name: [hours]}))[0])
    predicted_score = max(1.0, min(5.0, predicted_score))
    st.success(f"예상 수업 집중도는 약 {predicted_score:.1f}점입니다.")
    st.caption("예측값은 입력된 데이터의 경향을 이용한 참고용 추정치이며 실제 집중도를 확정하지 않습니다.")

st.header("6. 분석 데이터 다운로드")
csv_data = clean_data.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="분석 데이터 CSV 다운로드",
    data=csv_data,
    file_name="스마트폰_집중도_데이터.csv",
    mime="text/csv"
)
