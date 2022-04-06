# **로지스틱 회귀 & 확률적 경사 하강법**
- **head ( ) :** 처음 5개 행 출력 메소드
- **unique( )** : 판다스의 함수. 클래스들 출력해준다
  ``` python
  print(pd.unique(fish['Species']))

  => ['Bream', 'Roach', 'Whitefish', 'Parkki', 'Perch']
  ```
  
- **classes_ :** KneighborsClassifier에서 **알파벳 순으로 정렬**된 타깃값 classes_속성에 저장됨

- **predict_proba( )** : 클래스별 확률값 반환 메소드
- **round( ) :** 넘파이의 함수. 기본으로 소수점 첫째 자리에서 반올림. 그러나 decimals 이용해서 소수점 아래 자릿수 지정 가능
  ```python
  import numpy as np

  proba = kn.predict_proba(test_scaled[:5])

  print(np.round(proba, decimals = 4))
  # 소수점 네 번째 자리까지 표기
  => [0. 0. 0.6667 0. 0.3333 0. 0.]
  ```

## **로지스틱 회귀**
---
이름은 회귀이지만 분류 모델. 선형 방정식을 학습한다.

$$
z = a * (weight) + b *(length) + d * (height) + e *(width) + f
$$

**1.** a, b, c, d, e 는 가중치이다.

**2.** z는 음수 양수 상관없이 어떤 값도 가능하다. 


### **시그모이드 함수 (이중분류 때 사용)**
z가 아주 큰 양수일 때 1이, 아주 큰 음수일 때 0이 되게 해주는 함수

$$
\frac {1} {1+e^-z}
$$

시그모이드 함수 출력 > 0.5 => **양성 클래스**

시그모이드 함수 출력 <= 0.5 => **음성 클래스**

- **불리언 인덱싱** : True False 값 전달해서 넘파이 배열 행 선택
  ```python
  bream_smelt_indexes = (train_target == 'Bream') or (train_target == 'Smelt')

  train_bream_smelt = train_scaled[bream_smelt_indexes]

  target_bream_smelt = target_scaled[bream_smelt_indexes]

  => 도미와 빙어 값만 들어있게 된다
  ```

### **로지스틱 회귀 이진 분류 훈련 방법**
``` python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict_proba(train_bream_smelt[:5]))
=> [[0.99759855 0.00240145]]

# 첫번째 열이 음성 클래스(0)에 대한 값 _ 알파벳 순
# 두번째 열이 양성 클래스(1)에 대한 값 _ 알파벳 순

print(lr.coef_, lr.intercept_)
=> [[-0.4037798 -0.57620209 -0.66280298 -1.01290277 -0.73168947]] [-2.16155132]
# 각각 다 가중치(계수)이다.

decision = lr.decision_function(train_bream_smelt[:5])

print(decisions)
=> [-6.02927744 3.57123907 -5.26568906 -4.24321775 -6.0607117]
# decision_function()은 z값 구해주는 메소드

from scipy.special import expit

print(expit(decisions))
=> [0.00240145 0.97264817 0.00513928 0.01415798 0.00232731]

# expit(z) 는 시그모이드 함수
```

### **로지스틱 회귀 다중 분류 방법**

- **max_iter** : 반복횟수 지정. 기본값은 100. 에포크 횟수 지정

- 계수의 제곱을 규제( L2 규제라고도 부름)
- **C** : 규제 제어 매개변수. 작을수록 규제 커짐. 기본값 = 1
- 다중 분류는 클래스마다 z 값 하나씩 계산. 가장 높은 z값 출력하는 클래스가 예측 클래스

### **소프트맥스 함수(다중 분류에 이용)**
여러 개의 선형 방정식의 출력값을 0~1사이로 압축 후 전체 합이 1 이 되도록 만듦

$$
e_ sum = e^z1 + e^z2 + ... + e^z7
$$
$$
s1 = \frac{e^z1} {e_sum}, 
s7 = \frac{e^z7} {e_sum}
$$
7개 모두 합하면 1

```python
from scipy.special import softmax

proba = softmax(decision, axis = 1)
# axis = 1 은 각 행마다 소프트맥스 계산하라는 것

print(np.round(proba, decimals = 3))
=> [0. 0.014 0.841 0. 0.136 0.007 0.003]
```

## **점진적 학습**
---

- **점진적 학습 :** 앞서 훈련한 모델 버리지 않고 새로운 데이터에 대해서만 조금씩 더 훈련하는 것
- **확률적 :** '무작위하게', '랜덤하게'

### **확률적 경사 하강법**
1. 대표적인 점진적 학습 알고리즘
2. 샘플 하나씩 선택해서 경사 내려가기

- **에포크 :** 훈련 세트를 한 번 모두 사용하는 과정

### **미니배치 경사 하강법**
1. 무작위 여러개 샘플 사용해서 경사 내려가기

### **배치 경사 하강법**
1. 전체 샘플 사용해서 경사 내려가기

## **손실함수**
---
1. 어떤 문제에서 머신러닝 알고리즘이 얼마나 엉터리인지를 측정하는 기준. 

2. 작을수록 굿.
3. 미분 가능해야 함.

### **로지스틱 손실 함수 ( 이진 분류에 사용 )**
---

**양성 클래스일 때 손실 =** -log(예측 확률)

**음성 클래스일 때 손실 =** -log(1 - 예측 확률)

### **크로스엔트로피 손실 함수 ( 다중 분류에 사용 )**
---
- **SGDClassifier :** 사이킷런에서 확률적 경사 하강법 제공하는 대표적 분류용 클래스

- **loss :** 손실 함수 종류 지정

```python
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss = 'log', max_iter = 10, random_state = 42)

# loss = 'log'이므로 클래스마다 이진 분류 모델 만듦.
# 에포크 10회
```

### **훈련된 모델 sc추가로 훈련 시키기**
- **partial_fit()** 메소드 사용된다

```python
sc.partial_fit(train_scaled, train_target)
```
호출 할 때마다 1 에포크 이어서 훈련. 전달한 훈련 세트에서 1개씩 샘플 꺼내서 경하 하강법 단계 실행

### **에포크와 과대/과소적합**
에포크 횟수 적으면 => 과소적합

에포크 횟수 너무 많으면 => 과대적합

```python
sc = SGDClassifier(loss = 'log', max_iter = 100, tol = None, random_state = 42)
```
SGDClassifier는 일정 에포크 동안 성능 향상 없으면 더 훈련하지 않고 자동으로 멈춤.
- **tol :** 향상될 최솟값 지정