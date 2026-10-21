import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score
import io

st.set_page_config(page_title="Dynamic ML Explorer", layout="wide")

# ==========================================
# 1. 動態 Dataset 輸入 (加分條件 2)
# ==========================================
url = st.text_input("🔗 輸入 Dataset URL (加分條件):", "https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv")

@st.cache_data
def load_data(csv_url):
    return pd.read_csv(csv_url)

try:
    raw_df = load_data(url)
    
    # ==========================================
    # 🌟 根據 Dataset 內容，動態判定並標示主題
    # ==========================================
    cols_lower = [str(c).lower() for c in raw_df.columns]
    
    if 'r&d spend' in cols_lower or '50_startups' in url.lower() or 'profit' in cols_lower:
        theme_title = "📈 50 Startups 新創公司利潤預測 (Multiple Linear Regression)"
        theme_desc = "探索 50 Startups 資料集，分析研發、行銷等支出對利潤的影響，並進行特徵選擇與預測區間評估！"
        default_y_keyword = 'profit'
    elif 'enginesize' in cols_lower or 'carprice' in url.lower() or 'price' in cols_lower:
        theme_title = "預測探索器 "
        theme_desc = "探索車輛價格資料集，分析引擎大小、車重等特徵對車價的影響，並進行特徵選擇與預測區間評估！"
        default_y_keyword = 'price'
    else:
        theme_title = "📊 自訂資料集多元線性回歸探索器"
        theme_desc = "自動解析您輸入的資料集，進行多元線性迴歸、特徵選擇與模型預測評估！"
        default_y_keyword = None

    # 顯示動態主題
    st.title(theme_title)
    st.markdown(theme_desc)
    st.success(f"✅ 資料集載入成功！共 {len(raw_df)} 筆資料。")
    
    # 自動排除無法做線性迴歸的非數值字串欄位
    df = raw_df.select_dtypes(include=[np.number]).dropna()
    
    # 讓使用者動態選擇預測目標 Y (根據主題自動尋找預設值)
    cols = df.columns.tolist()
    default_y_index = len(cols) - 1
    if default_y_keyword:
        for i, col in enumerate(cols):
            if default_y_keyword in col.lower():
                default_y_index = i
                break
                
    target_col = st.selectbox("🎯 請選擇預測目標 (Target Y):", cols, index=default_y_index)
    
    # ==========================================
    # 2. 問題切換下拉選單 (加分條件 1)
    # ==========================================
    problem = st.selectbox("📌 SELECT PROBLEM VIEW (加分條件):", 
                           ["Problem 1: Feature Selection & Model Evaluation (特徵選擇與評估)", 
                            "Problem 2: Prediction Intervals & Live Predictor (區間圖與即時預測)"])
    st.markdown("---")
    
    # 資料前處理
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 執行 RFE 特徵選擇
    max_k = min(10, len(X.columns)) 
    results = []
    best_r2 = -float('inf')
    best_k = 1
    best_features = []
    
    for k in range(1, max_k + 1):
        rfe = RFE(estimator=LinearRegression(), n_features_to_select=k)
        rfe.fit(X_train, y_train)
        sel_cols = X.columns[rfe.support_].tolist()
        
        model = LinearRegression()
        model.fit(X_train[sel_cols], y_train)
        preds = model.predict(X_test[sel_cols])
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results.append({"k": k, "RMSE": rmse, "R2": r2, "features": sel_cols})
        
        if r2 > best_r2:
            best_r2 = r2
            best_k = k
            best_features = sel_cols
            
    res_df = pd.DataFrame(results)

    # ==========================================
    # Problem 1: 特徵選擇與模型評估
    # ==========================================
    if problem.startswith("Problem 1"):
        st.subheader("🧠 特徵選擇與模型評估指標")
        col1, col2, col3 = st.columns(3)
        col1.metric("最佳特徵數量 (Best k)", f"{best_k} / {len(X.columns)}")
        col2.metric("最佳 Test RMSE", f"{res_df[res_df['k']==best_k]['RMSE'].values[0]:,.2f}")
        col3.metric("最佳 R² Score", f"{best_r2:.4f}")
        
        st.info("**最佳特徵組合:** " + ", ".join(best_features))
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📉 RMSE 隨特徵數量變化")
            st.line_chart(res_df.set_index('k')['RMSE'])
        with c2:
            st.markdown("#### 📈 R² 隨特徵數量變化")
            st.line_chart(res_df.set_index('k')['R2'])
            
    # ==========================================
    # Problem 2: 預測區間圖與即時預測
    # ==========================================
    else:
        st.subheader("📈 迴歸方程式、預測區間與即時預測")
        
        # 使用 statsmodels 計算係數
        X_train_sm = sm.add_constant(X_train[best_features])
        model_sm = sm.OLS(y_train, X_train_sm).fit()
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📊 Regression Coefficients")
            st.dataframe(model_sm.summary2().tables[1][['Coef.', 'P>|t|']], use_container_width=True)
            
            # 繪製預測區間圖
            st.markdown("#### 🎯 預測區間圖 (Actual vs Predicted)")
            X_test_sm = sm.add_constant(X_test[best_features], has_constant='add')
            pred_res = model_sm.get_prediction(X_test_sm).summary_frame(alpha=0.05)
            
            plot_df = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': pred_res['mean'].values,
                'CI_Lower': pred_res['mean_ci_lower'].values,
                'CI_Upper': pred_res['mean_ci_upper'].values
            }).sort_values('Actual').reset_index(drop=True)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(plot_df.index, plot_df['Actual'], 'ko', label='Actual Data')
            ax.plot(plot_df.index, plot_df['Predicted'], 'r-', linewidth=2, label='Predicted Line')
            ax.fill_between(plot_df.index, plot_df['CI_Lower'], plot_df['CI_Upper'], color='red', alpha=0.2, label='95% Confidence Interval')
            ax.set_xlabel("Test Samples (Sorted by Actual Value)")
            ax.set_ylabel(target_col)
            ax.legend()
            
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
            st.image(buf, use_container_width=True)
            plt.close(fig)
            
        with c2:
            st.markdown("#### 🎛️ 即時預測器 (Live Predictor)")
            st.write("根據最佳特徵，拖曳滑桿進行即時預測：")
            
            coef_dict = model_sm.params.to_dict()
            intercept = coef_dict.get('const', 0.0)
            predicted_value = intercept
            
            for f in best_features:
                min_val = float(X[f].min())
                max_val = float(X[f].max())
                mean_val = float(X[f].mean())
                
                if set(X[f].dropna().unique()).issubset({0, 1}):
                    user_input = st.selectbox(f"選擇 {f}", [0, 1])
                else:
                    user_input = st.slider(f"輸入 {f}", min_value=min_val, max_value=max_val, value=mean_val)
                    
                predicted_value += (coef_dict[f] * user_input)
            
            st.success(f"### 預測 {target_col} 值:\n# **{predicted_value:,.2f}**")

except Exception as e:
    st.error(f"無法載入或解析資料集，錯誤訊息: {e}")