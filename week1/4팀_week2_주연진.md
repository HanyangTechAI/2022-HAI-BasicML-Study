# **회귀 알고리즘과 모델 규제**
## **K-최근접 이웃 회귀**
---
* 주변 샘플 k개를 선택해 타깃값을 평균내서 타깃값을 예측하는 방법
* 사이킷런 훈련 세트는 2차원 배열이어야 함
```python
test_array = np.array([1, 2, 3, 4])

test_array = test_array.reshape(2, 2)

print(test_array.shape)
```
1. 처음 [1, 2, 3, 4] 의 경우 크기는 (4, )이다. ( 1차원 배열 )

2. reshape( )를 사용하면 크기가 (2, 2)로 변경 된다. ( 2차원 배열 ) 

3. reshape( -1, 1 )을 하면 첫 번째 크기를 나머지 원소 개수로 채우고, 두 번째 크기는 1이 된다.

* **KNeighborsRegressor** : k - 최근접 이웃 회귀 알고리즘 구현 클래스
```python
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()

kn.fit(train_input, train_target)
```

## **결정계수 ( $R^2$ )** 
---
```python
print(knr.score(test_input, test_target))
```
결과값으로 0.992809406100639가 나온다.

* **결정계수 ( $R^2$ )** : 위 결과값

$$
R^2 = 1 - \frac{(타깃 - 예측)^2의 합 }{ (타깃 - 평균)^2의 합}
$$
이와 같이 결정계수를 구할 수 있다. 
> * 평균 = 타깃 값들의 평균
> * 결정계수는 분산 이용한 모델평가지표
> * 독립변수로 설명할 수 있는 분산 / 전체분산 과 같다
> * 독립변수들이 종속변수들을 얼마나 설명하는가 
> * 1에 가까울수록 좋은 모델

## **타깃과 예측한 값 차이 구하기**
---
* **mean_absolute_error :** 타깃과 예측의 절댓값 오차 평균 반환
```python
from sklearn.metrics import mean_absolute_error

test_prediction = knr.predict(test_input)

mae = mean_absolute_error(test_target, test_prediction)
```
다음과 같이 사용한다.

## **과대적합 vs 과소적합** 
---
### **과대적합**
* 훈련 세트에서 점수가 굉장히 좋은데, 테스트 세트에서 점수가 굉장히 나쁜 경우 
* 훈련 세트에만 잘 맞는 모델

### **과소적합**
* 훈련 세트보다 테스트 세트 점수가 더 놓은 경우
* 두 점수 모두 낮은 경우


### **과소적합이 일어나는 이유 :** 
* 모델이 너무 단순하여 훈련 세트에 적절히 훈련되지 않은 경우
* 훈련 세트와 테스트 세트의 크기가 매우 작기 때문

### **과소적합 해결 방법**
모델을 조금 더 복잡하게 만들면 된다. K-최근접 이웃 알고리즘의 경우 K를 줄이면 된다.

## **선형회귀**
___
* 사이킷럿에 LinearRegression 클래스에서 선형 회귀 알고리즘 사용
```python
from sklenar.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(train_input, train_target)

print(lr.predict([[50]]))
```
다음과 같이 사용한다.

 * **coef_ :** 기울기 ( 계수, 또는 가중치라고도 불림 )
 * **intercept_ :** y절편

 ## **다항회귀**
 ---
 * 최적의 곡선 찾기
 * 2차 방정식 그래프 그리려면 길이 제곱이 훈련에 추가 되어야 함
```python
train_poly = np.column_stack((train_input ** 2, train_input)) 
test_poly = np.column_stack((test_input ** 2, test_input))
```
* 어떤 그래프를 훈련하든지, 타깃값은 그대로 사용한다.

```python
lr = LinearRegression()
lr.fit(train_poly, train_target)

print([[50 ** 2, 50]])
```
## **다중 회귀**
---
* 여러 개의 **특성**을 사용한 선형 회귀

* **특성 공학 :** 기존의 특성을 사용해 새로운 특성을 뽑아내는 작업
   > ex) '농어 길이' x '농어 높이'를 새로운 특성으로 만들기

## **판다스**
---
* 데이터 분석 라이브러리
* **넘파이와 달리** 인터넷에서 데이터 바로 다운로드해서 사용 가능
* 데이터 프라임 넘파이 배열로 바꾸는 거 가능
* read_csv( '주소' ) 로 데이터 프라임 만들고, to_numpy() 

```python
import pandas as pd

df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)
# [[8.4, 2.11, 1.41], 
#  [13.7, 3.53, 2. ]]
# 위와 같이 여러 특성을 포함한 numpy로 바뀐다.
```

## **사이킷런 변환기**
---
* **변환기:** 특성을 만들거나 전처리해주는 클래스
  > ex) PolynomialFeatures
* **include_bias = False :** transform 할 때 1을 자동으로 추가하는 것을 방지
* **transform() :** 특성 각자 그대로, 각자 제곱한 값, 서로서로 곱한 값 모두 포함
* **polynomialFeatures(degree = 5) :** 5제곱까지 특성 만듦

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures()

poly.fit([[2, 3]]) #훈련 해야 변환 가능
print(poly.transform([[2,3]]))
# [[1,2,3,4,6,9]] 가 나옴.
# 2개 특성이 6개로 바뀜.
```

## **규제**
---
* 훈련 세트에 과대적합되지 않도록 만드는 것
* 선형 회귀 모델의 경우 특성에 곱해지는 계수 작게 만들기
* 선형 회귀 모델에 규제 적용할 때, 계수 깂 크기가 특성마다 많이 다르면 공정한 제어 불가능하므로 정규화 필요
* **StandardScaler :** 특성 표준점수로 바꿔주는 변환기

```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_poly)

train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)
```

### **Alpha**
* Alpha는 매개변수로 규제 강도 조절

* Alpha 값이 **크면** 규제 커지고 **과소적합** 유도
* Alpha 값이 **작으면** 규제 작아지고 **과대적합** 유도
* **하이퍼파라미터 :** 모델이 학습할 수 없고 사람이 알려줘야 하는 파라미터


### **릿지 회귀**
* 선형 회귀 모델에 규제를 추가한 모델
* 계수를 제곱한 값을 기준으로 규제
```python
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(train_scaled, test_scaled)
```
이와 같이 사용한다.

### **라쏘 회귀**
* 계수의 절댓값을 기준으로 규제 적용
* 계수 크기 0으로 만들기 가능
```python
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(train_scaled, test_scaled)
```







