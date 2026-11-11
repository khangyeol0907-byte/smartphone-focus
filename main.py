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
    "실제 설문조사 결과가 아니므로, 수행평가에서는 설문조사로 수집한 데이터로 바꿔서 분석하세요."
)

x_name = "스마트폰 사용 시간(시간)"
y_name = "수업 집중도(1~5점)"
required_columns = [x_name, y_name]

sample_data = pd.DataFrame({
    x_name: [1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7],
    y_name: [5, 4, 5, 4, 3, 4, 3, 2, 3, 1]
})

st.header("1. 데이터 입력")
st.write("표의 숫자를 직접 수정하거나, 설문조사 결과가 담긴 CSV 파일을 업로드할 수 있습니다.")

uploaded_file = st.file_uploader("설문조사 CSV 파일 업로드", type=["csv"])

source_name = "직접 입력한 데이터"
if uploaded_file is not None:
    try:
        uploaded_data = pd.read_csv(uploaded_file)
        # 이전 파일에서 사용했던 짧은 열 이름도 호환되도록 변환
        uploaded_data = uploaded_data.rename(columns={
            "사용 시간": x_name,
            "집중도": y_name
        })

        missing_columns = [col for col in required_columns if col not in uploaded_data.columns]
        if missing_columns:
            st.error(
                "CSV 파일의 열 이름이 올바르지 않습니다. "
                f"필요한 열 이름은 '{x_name}', '{y_name}'입니다. "
                f"빠진 열: {', '.join(missing_columns)}"
            )
            st.stop()

        extra_columns = [col for col in uploaded_data.columns if col not in required_columns]
        if extra_columns:
            st.warning(
                "분석에 필요한 두 열 이외의 열은 분석에서 제외합니다: "
                + ", ".join(map(str, extra_columns))
            )

        data = uploaded_data[required_columns].copy()
        source_name = "업로드한 CSV 데이터"

    except Exception as error:
        st.error(f"CSV 파일을 읽는 중 문제가 발생했습니다: {error}")
        st.stop()
else:
    data = sample_data.copy()

st.caption(f"현재 데이터 출처: {source_name}")

st.write("아래 표에서 값을 직접 수정할 수 있습니다. 사용 시간은 0~24, 집중도는 1~5 사이의 정수여야 합니다.")
edited_data = st.data_editor(
    data,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        x_name: st.column_config.NumberColumn(
            x_name, min_value=0.0, max_value=24.0, step=0.5,
            help="0시간 이상 24시간 이하로 입력하세요."
        ),
        y_name: st.column_config.NumberColumn(
            y_name, min_value=1, max_value=5, step=1,
            help="1점부터 5점까지의 정수로 입력하세요."
        )
    }
)

# 데이터 검증: 빈 행은 사용자가 표에 추가한 미입력 행일 수 있으므로
# 완전히 비어 있는 행은 오류 개수에 포함하지 않고 분석에서도 제외한다.
raw = edited_data[required_columns].copy()
row_has_any_value = raw.notna().any(axis=1)
raw_nonempty = raw.loc[row_has_any_value].copy()

numeric_x = pd.to_numeric(raw_nonempty[x_name], errors="coerce")
numeric_y = pd.to_numeric(raw_nonempty[y_name], errors="coerce")

valid_x = numeric_x.notna() & numeric_x.between(0, 24)
valid_y_range = numeric_y.notna() & numeric_y.between(1, 5)
valid_y_integer = numeric_y.notna() & np.isclose(numeric_y.fillna(0) % 1, 0)
valid_rows = valid_x & valid_y_range & valid_y_integer

invalid_count = int((~valid_rows).sum())
clean_data = pd.DataFrame({
    x_name: numeric_x.loc[valid_rows].astype(float),
    y_name: numeric_y.loc[valid_rows].astype(int)
}).reset_index(drop=True)

st.subheader("데이터 검증 결과")
col_a, col_b, col_c = st.columns(3)
col_a.metric("입력된 데이터 행", f"{len(raw_nonempty)}개")
col_b.metric("정상 데이터", f"{len(clean_data)}개")
col_c.metric("제외된 잘못된 데이터", f"{invalid_count}개")

if invalid_count > 0:
    st.warning(
        f"잘못된 데이터 {invalid_count}개를 확인했습니다. "
        "빈 값, 숫자가 아닌 값, 사용 시간 범위(0~24시간) 초과, "
        "집중도 범위(1~5점) 초과 또는 집중도 소수점 값은 분석에서 제외했습니다."
    )
if len(raw_nonempty) == 0:
    st.info("분석할 데이터가 없습니다. 표에 값을 입력하거나 CSV 파일을 업로드하세요.")
    st.stop()

st.caption("분석에는 검증을 통과한 정상 데이터만 사용합니다.")

st.header("2. 데이터 소개")
metric1, metric2, metric3 = st.columns(3)
metric1.metric("정상 데이터 수", f"{len(clean_data)}명")
if len(clean_data) > 0:
    metric2.metric("평균 스마트폰 사용 시간", f"{clean_data[x_name].mean():.1f}시간")
    metric3.metric("평균 수업 집중도", f"{clean_data[y_name].mean():.2f}점")
else:
    metric2.metric("평균 스마트폰 사용 시간", "-")
    metric3.metric("평균 수업 집중도", "-")

st.subheader("정상적으로 입력된 데이터")
st.dataframe(clean_data, use_container_width=True)

if len(clean_data) < 3:
    st.warning(
        f"정상 데이터가 {len(clean_data)}개입니다. "
        "기초 통계는 확인할 수 있지만, 안정적인 선형회귀 분석을 위해서는 "
        "정상 데이터가 최소 3개 필요합니다."
    )
    st.stop()

if clean_data[x_name].nunique() < 2:
    st.warning(
        "스마트폰 사용 시간이 모두 같아 회귀선을 계산할 수 없습니다. "
        "서로 다른 사용 시간의 데이터가 2개 이상 포함되도록 입력하세요."
    )
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
r2 = r2_score(y, predictions) if y.nunique() > 1 else 0.0

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

chart_data = clean_data.copy()
chart_data["회귀선 예측값"] = predictions

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
if y.nunique() == 1:
    st.caption("참고: 집중도 값이 모두 같아 R²는 해석에 주의가 필요합니다.")

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
