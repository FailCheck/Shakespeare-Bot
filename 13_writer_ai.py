import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(0)

# ---------------------------------------------------------
# 2. 데이터 준비: 셰익스피어의 소네트 18번 (일부분)
# ---------------------------------------------------------
sentence = ("Shall I compare thee to a summer's day? "
            "Thou art more lovely and more temperate: "
            "Rough winds do shake the darling buds of May, "
            "And summer's lease hath all too short a date:")

print(f"학습할 문장 길이: {len(sentence)}글자")

# 문자 집합 만들기 (사전 제작)
char_set = sorted(list(set(sentence)))
char_dic = {c: i for i, c in enumerate(char_set)}
dic_size = len(char_dic) # 문자 종류의 개수

print(f"문자 사전 크기: {dic_size}개")

# 데이터 가공 (Hyperparameters)
hidden_size = 128   # 뇌세포 개수 (많을수록 똑똑함)
sequence_length = 10 # AI에게 한 번에 보여줄 글자 수 (10글자 보고 다음 글자 맞히기)
learning_rate = 0.01

# 데이터셋 만들기 (Sliding Window)
# 예: "Shall I co" -> "m", "hall I com" -> "p" ... 옆으로 한 칸씩 밀면서 문제집을 만듭니다.
x_data = []
y_data = []

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i : i + sequence_length]
    y_str = sentence[i + 1 : i + sequence_length + 1]
    
    x_data.append([char_dic[c] for c in x_str])
    y_data.append([char_dic[c] for c in y_str])

# 원-핫 인코딩 & 텐서 변환
x_one_hot = [np.eye(dic_size)[x] for x in x_data]
X = torch.tensor(x_one_hot, dtype=torch.float32).to(device)
Y = torch.tensor(y_data, dtype=torch.long).to(device)

# ---------------------------------------------------------
# 3. 모델 설계 (LSTM: 긴 기억력을 가진 신경망)
# ---------------------------------------------------------
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(Net, self).__init__()
        # RNN 대신 LSTM을 씁니다! (업그레이드)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim, bias=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

model = Net(dic_size, hidden_size, 1).to(device) # 층을 2개 쌓아서 더 깊게!

# 4. 학습 시작
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("\n=== AI 작가 데뷔 준비 중 (학습 시작) ===")
epochs = 3000 # 100번 반복 학습

for i in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    
    # 오차 계산
    loss = criterion(outputs.view(-1, dic_size), Y.view(-1))
    
    # 미분 및 업데이트
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        # 현재까지 배운 걸로 아무말 대잔치 해보기
        results = outputs.argmax(dim=2)
        predict_str = ""
        for j, result in enumerate(results):
            if j == 0: # 첫 번째 문장은 다 가져오고
                predict_str += ''.join([char_set[t] for t in result])
            else: # 그 뒤부터는 마지막 글자만 이어 붙임
                predict_str += char_set[result[-1]]

        print(f"Epoch {i}: {loss.item():.4f}")
        print(f"생성된 문장: {predict_str}\n")

print("---------------------------------")
print("✅ 최종 결과 (원본 vs AI 작문):")
print(f"원본: {sentence}")
print(f"AI  : {predict_str}")