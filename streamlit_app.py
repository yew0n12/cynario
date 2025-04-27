# streamlit_app.py

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sentence_transformers import SentenceTransformer
from pathlib import Path

# 너가 만든 클래스 불러오기
# ChatParser, SentimentInteractionAnalyzer, CyberbullyingStatementGenerator 클래스는 이미 다 작성되어 있어야 함
from streamlit_python_code import ChatParser, SentimentInteractionAnalyzer, CyberbullyingStatementGenerator

def main():
    st.title("사이버불링 자동 분석 시스템")

    # 파일 업로드
    uploaded_file = st.file_uploader("카카오톡 TXT 파일을 업로드하세요", type=['txt'])

    if uploaded_file is not None:
        # 임시로 파일 저장
        temp_path = "uploaded_chat.txt"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 한글 폰트 설정
        font_path = 'C:/Windows/Fonts/NanumGothic.ttf'
        if Path(font_path).exists():
            fontprop = fm.FontProperties(fname=font_path)
            plt.rc('font', family=fontprop.get_name())
        else:
            st.warning("폰트 파일을 찾을 수 없습니다. 기본 폰트로 진행합니다.")

        plt.rcParams['axes.unicode_minus'] = False

        # 모델 불러오기
        with st.spinner("모델 로딩 중..."):
            similarity_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

        # 분석 시작
        with st.spinner("채팅 파일 분석 중..."):
            parser = ChatParser()
            parser.parse_file(temp_path)

            analyzer = SentimentInteractionAnalyzer(parser, similarity_model)
            analyzer.analyze()
            analyzer.save_persistent_cases_to_csv('persistent_cases.csv')
            analyzer.visualize_persistent_cases('persistent_cases_graph.png', 'persistent_cases_graph.html')

            statement_generator = CyberbullyingStatementGenerator(
                csv_path='persistent_cases.csv',
                png_path='persistent_cases_graph.png',
                html_path='persistent_cases_graph.html'
            )
            statement_generator.generate_statement("cyberbullying_statement.md")
            statement_generator.deliver_to_victim()

        st.success("분석 완료! 결과 파일을 다운로드할 수 있습니다.")

        # 다운로드 버튼들
        with open("persistent_cases.csv", "rb") as f:
            st.download_button(
                label="📄 지속적 공격 사례 CSV 다운로드",
                data=f,
                file_name="persistent_cases.csv",
                mime="text/csv"
            )

        with open("persistent_cases_graph.png", "rb") as f:
            st.download_button(
                label="📊 지속적 공격 관계 그래프(PNG) 다운로드",
                data=f,
                file_name="persistent_cases_graph.png",
                mime="image/png"
            )

        with open("persistent_cases_graph.html", "rb") as f:
            st.download_button(
                label="🌐 지속적 공격 관계 그래프(HTML) 다운로드",
                data=f,
                file_name="persistent_cases_graph.html",
                mime="text/html"
            )

        with open("cyberbullying_statement.md", "rb") as f:
            st.download_button(
                label="📝 진술서(MD 파일) 다운로드",
                data=f,
                file_name="cyberbullying_statement.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    main()
