import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# 1. AI 모델 구조
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim, bias=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


# 2. 저장된 뇌(shakespeare.pt) 불러오기 

@st.cache_resource 
# 웹사이트가 새로고침 될 때마다 모델을 다시 로딩하지 않게 함 (속도 향상)

def load_model():
    # 파일이 있는지 확인
    try:
        # 저장된 딕셔너리 불러오기
        checkpoint = torch.load('shakespeare.pt', map_location=torch.device('cpu'))
        
        # 족보(사전) 복구
        loaded_chars = checkpoint['chars']
        loaded_char_dic = {c: i for i, c in enumerate(loaded_chars)}
        dic_size = checkpoint['dic_size']
        hidden_size = checkpoint['hidden_size']
        
        # 모델 뼈대 만들고 가중치 끼우기
        loaded_model = Net(dic_size, hidden_size, 1) # 저장할때 층(layer) 1개였는지 2개였는지 기억하세요! (아까 수정했으면 1)
        loaded_model.load_state_dict(checkpoint['model'])
        loaded_model.eval() # 평가 모드
        
        return loaded_model, loaded_char_dic, loaded_chars, dic_size
        
    except FileNotFoundError:
        return None, None, None, None

# 모델 로드 실행
model, char_dic, char_set, dic_size = load_model()

# ---------------------------------------------------------
# 3. 웹사이트 화면 구성 (UI)
# ---------------------------------------------------------
st.title("전현우의 첫 인공지능 웹사이트(Beta)")
st.caption("처음이긴 한데, 성공했쥬?")

if model is None:
    st.error("❌ 오류: 'shakespeare.pt' 파일이 없습니다. 13번 코드를 먼저 실행하세요!")
else:
    # 사용자 입력
    user_input = st.text_input("영어 단어를 입력하세요 (예: Shall)")

    if st.button("시 작성하기 (Write Poem)"):
        with st.spinner('셰익스피어 문단 제작중...'):
            
            # --- AI 예측 로직 시작 ---
            input_str = user_input
            
            # 1) 입력된 글자를 숫자로 변환 (전처리)
            try:
                x_input = [char_dic[c] for c in input_str]
                x_one_hot = [np.eye(dic_size)[x] for x in x_input]
                X = torch.tensor(x_one_hot, dtype=torch.float32).unsqueeze(0)
                
                # 2) 예측 시작
                predict_str = input_str
                
                # 50글자 정도 더 써보라고 시키기
                for i in range(50):
                    outputs = model(X)
                    
                    # 가장 확률 높은 다음 글자 선택
                    result = outputs.data.numpy().argmax(axis=2)
                    next_char_idx = result[0][-1] # 맨 마지막 글자의 예측값
                    next_char = char_set[next_char_idx]
                    
                    predict_str += next_char
                    
                    # 다음 입력을 위해 데이터 업데이트 (Sliding)
                    # 현재 예측한 글자를 다음 스텝의 입력으로 씀
                    next_one_hot = np.eye(dic_size)[next_char_idx]
                    next_tensor = torch.tensor(next_one_hot, dtype=torch.float32).view(1, 1, -1)
                    X = torch.cat([X, next_tensor], dim=1)

                st.success("작성 완료!")
                st.markdown("### 🖋️ AI의 창작물:")
                st.info(predict_str)
                
            except KeyError:
                st.error("⚠️ 죄송합니다. AI가 아직 배우지 못한 글자가 포함되어 있어요! (대소문자 등을 확인해주세요)")


