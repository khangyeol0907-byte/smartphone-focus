
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="소셜미디어 사용과 학습 집중도 분석",
    page_icon="📱",
    layout="wide"
)

st.title("📱 소셜미디어 사용과 학습 집중도 분석")
st.write(
    "학생들의 소셜미디어 사용 습관과 학습 집중도 및 학업 생산성의 관계를 분석합니다."
)

st.info(
    "CSV 파일을 업로드하면 실제 데이터로 분석합니다. "
    "파일이 없을 때는 기능 확인용 예시 데이터가 표시됩니다."
)

# 예시 데이터: 실제 조사 결과가 아님
sample_data = pd.DataFrame({
    "SocialMediaHours": [1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7],
    "FocusReduction": [0, 1, 0, 1, 1, 2, 2, 2, 3, 3],
    "Procrastination": [0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
    "UsageDuringStudy": [0, 1, 0, 1, 1, 2, 2, 2, 3, 3]
})

# 1. 데이터 입력
st.header("1. 데이터 입력")

uploaded_file = st.file_uploader(
    "외부 설문조사 CSV 파일 업로드",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        try:
            data = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            data = pd.read_csv(uploaded_file, encoding="cp949")

        source_name = "외부 CSV 데이터"

    except Exception as e:
        st.error(f"파일을 읽을 수 없습니다: {e}")
        st.stop()
else:
    data = sample_data.copy()
    source_name = "기능 확인용 예시 데이터"

st.caption(f"현재 데이터 출처: {source_name}")

if data.empty:
    st.warning("데이터가 비어 있습니다.")
    st.stop()

st.write("데이터 미리보기")
st.dataframe(data.head(10), use_container_width=True)

st.metric("전체 데이터 행 수", f"{len(data)}개")
st.metric("전체 변수 수", f"{len(data.columns)}개")

# 2. 변수 선택
st.header("2. 분석 변수 선택")

numeric_columns = data.select_dtypes(include=np.number).columns.tolist()

if len(numeric_columns) < 2:
    st.error("분석을 위해 숫자로 된 변수가 최소 2개 필요합니다.")
    st.stop()

default_x = (
    "SocialMediaHours"
    if "SocialMediaHours" in numeric_columns
    else numeric_columns[0]
)

default_y = (
    "FocusReduction"
    if "FocusReduction" in numeric_columns
    else numeric_columns[1]
)

col1, col2 = st.columns(2)

with col1:
    x_name = st.selectbox(
        "독립변수 (X)",
        numeric_columns,
        index=numeric_columns.index(default_x)
    )

with col2:
    y_options = [c for c in numeric_columns if c != x_name]
    y_default = y_options.index(default_y) if default_y in y_options else 0

    y_name = st.selectbox(
        "종속변수 (Y)",
        y_options,
        index=y_default
    )

st.caption(
    "소셜미디어 사용 시간과 집중력 저하를 기본 분석 변수로 설정했습니다. "
    "다른 숫자형 변수를 선택해 관계를 비교할 수도 있습니다."
)

# 3. 데이터 검증
st.header("3. 데이터 검증")

analysis_data = data[[x_name, y_name]].copy()

for col in [x_name, y_name]:
    analysis_data[col] = pd.to_numeric(
        analysis_data[col], errors="coerce"
    )

original_count = len(analysis_data)

analysis_data = analysis_data.replace(
    [np.inf, -np.inf], np.nan
)

clean_data = analysis_data.dropna().copy()

removed_count = original_count - len(clean_data)

a, b, c = st.columns(3)
a.metric("원본 데이터", f"{original_count}개")
b.metric("정상 데이터", f"{len(clean_data)}개")
c.metric("제외된 데이터", f"{removed_count}개")

if removed_count > 0:
    st.warning(
        f"결측값이나 숫자가 아닌 값 등 {removed_count}개 행을 제외했습니다."
    )

if len(clean_data) < 3:
    st.warning("분석을 위해 정상 데이터가 최소 3개 필요합니다.")
    st.stop()

st.dataframe(clean_data, use_container_width=True)

# 4. 기초 통계
st.header("4. 기초 통계 분석")

stats = clean_data.describe().T
st.dataframe(stats, use_container_width=True)

m1, m2, m3 = st.columns(3)
m1.metric(f"{x_name} 평균", f"{clean_data[x_name].mean():.2f}")
m2.metric(f"{y_name} 평균", f"{clean_data[y_name].mean():.2f}")
m3.metric("분석 데이터 수", f"{len(clean_data)}개")

# 5. 데이터 시각화
st.header("5. 데이터 시각화")

fig1 = px.histogram(
    clean_data,
    x=x_name,
    title=f"{x_name} 분포",
    marginal="box"
)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.scatter(
    clean_data,
    x=x_name,
    y=y_name,
    title=f"{x_name}과 {y_name}의 관계",
    trendline="ols"
)
st.plotly_chart(fig2, use_container_width=True)

# 6. 상관관계 분석
st.header("6. 상관관계 분석")

correlation = clean_data[x_name].corr(clean_data[y_name])

st.metric("피어슨 상관계수", f"{correlation:.3f}")

if correlation < -0.3:
    st.write("두 변수 사이에 음의 상관관계가 나타납니다.")
elif correlation > 0.3:
    st.write("두 변수 사이에 양의 상관관계가 나타납니다.")
else:
    st.write("두 변수 사이의 선형 상관관계가 약하게 나타납니다.")

st.caption(
    "상관관계는 두 변수의 관련성을 나타내며, 한 변수가 다른 변수의 원인이라는 뜻은 아닙니다."
)

# 7. 선형회귀 분석
st.header("7. 선형회귀 분석")

if clean_data[x_name].nunique() < 2:
    st.warning("독립변수의 값이 모두 같아 회귀분석을 진행할 수 없습니다.")
    st.stop()

X = clean_data[[x_name]]
y = clean_data[y_name]

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

slope = float(model.coef_[0])
intercept = float(model.intercept_)

mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))

r2 = (
    r2_score(y, predictions)
    if y.nunique() > 1
    else None
)

st.write(
    f"**회귀식:** {y_name} = {intercept:.3f} + "
    f"({slope:.3f} × {x_name})"
)

if slope < 0:
    st.write(
        f"{x_name}이 1 증가할 때 {y_name}의 예측값은 "
        f"약 {abs(slope):.3f} 감소하는 경향이 있습니다."
    )
elif slope > 0:
    st.write(
        f"{x_name}이 1 증가할 때 {y_name}의 예측값은 "
        f"약 {slope:.3f} 증가하는 경향이 있습니다."
    )
else:
    st.write("두 변수 사이에 뚜렷한 선형 변화가 나타나지 않습니다.")

# 8. 모델 평가
st.header("8. 모델 평가")

r1, r2_col, r3 = st.columns(3)

r1.metric(
    "결정계수 R²",
    f"{r2:.3f}" if r2 is not None else "계산 불가"
)
r2_col.metric("평균 절대 오차 MAE", f"{mae:.3f}")
r3.metric("평균 제곱근 오차 RMSE", f"{rmse:.3f}")

st.write(
    "R²는 모델의 데이터 설명력을 나타냅니다. "
    "MAE와 RMSE는 예측값과 실제값의 차이를 나타내며, "
    "일반적으로 값이 작을수록 예측 오차가 작습니다."
)

st.warning(
    "현재 평가는 모델을 학습한 데이터에 대한 평가입니다. "
    "새로운 데이터에 대한 성능을 보장하지 않습니다."
)

# 9. 새로운 값 예측
st.header("9. 새로운 값으로 결과 예측")

minimum = float(clean_data[x_name].min())
maximum = float(clean_data[x_name].max())

if minimum < maximum:
    new_x = st.number_input(
        f"{x_name} 입력",
        min_value=minimum,
        max_value=maximum,
        value=float(np.clip(
            clean_data[x_name].mean(), minimum, maximum
        ))
    )

    if st.button("예측 결과 확인"):
        new_prediction = float(
            model.predict(pd.DataFrame({x_name: [new_x]}))[0]
        )

        st.success(
            f"예측된 {y_name}: {new_prediction:.3f}"
        )

        st.caption(
            "예측 결과는 회귀모델에 따른 추정값이며 실제 결과와 다를 수 있습니다."
        )
else:
    st.info("예측을 위해 서로 다른 독립변수 값이 필요합니다.")

# 10. 다운로드
st.header("10. 분석 데이터 다운로드")

csv_data = clean_data.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    "정제된 분석 데이터 CSV 다운로드",
    data=csv_data,
    file_name="분석_정제데이터.csv",
    mime="text/csv"
)

st.caption(
    "※ 예시 데이터로 표시된 분석 결과는 실제 설문조사 결과가 아닙니다."
)
