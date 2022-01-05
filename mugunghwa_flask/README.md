## 달고나 게임 Read Me

<p align="center">
    
</p>
<h4 align="center">드라마 '오징어 게임' 속 컨텐츠를 직접 체험할 수 있는 AI 플랫폼</h4>
<p align="center">
  <a href="#tutorial">Tutorial</a></a> • 
  <a href="#features">Features</a> •  
  <a href="#system-structure">System Structures</a> •
  <a href="#Files">Files</a> • 
  <a href="#contributor">Contributors</a> • 
  <a href="#license">License</a>
</p>
<p align="center">
    이 프로젝트는 2022 KT 하반기 인턴교육 과정 중 진행되었습니다. <br/>
    이 프로젝트는 상업적인 목적이 포함되어 있지 않습니다. 
    이 프로젝트는 팀 '우린깐부잖어'에 의해 개발되었습니다.<br/>
    해당 레포는 'AI 오징어 게임'의 웹 페이지 코드를 저장하고 있습니다.      
</p>




## Features

<p align="center">
    <h5>1. 객체 인식</h5>
    <h5>2. 신체 landmark 탐지</h5>
    <h5>3. 신체 움직임 frame 단위 탐지</h5>
	<h5>4. 게임 동작 시간 설정</h5>
	<h5>5. 설정된 시간에 맞게 frame count</h5>
	<h5>6. 탐지된 landmark 간 각도 계산</h5>
	<h5>7. 계산된 각도 값을 기준으로 목표 포즈에 맞는지 비교, 구별</h5>
	<h5>8. frame 단위로 구분지어, 전/후 frame 비교, threshold 값 계산</h5>
	<h5>9. 움직임 여부와 포즈 일치 여부를 바탕으로 점수 계산(백분율)</h5>
	<h5>10. 점수 표시</h5>
    <h5>11. 게임 종료 조건 설정</h5>
    <h5>12. 게임 반복을 위한 초기화</h5>
    <h5>13. 게임 로딩 API</h5>
    <h5>14. 게임 결과 API</h5>
</p>


## Files
<p align="center">
    <h5>1. images : 미션 포즈 이미지 저장 폴더</h5>
    <h5>2. templates : 테스트 페이지 저장 폴더</h5>
    <h5>3. util.py : 관련 라이브러리 및 클래스, 함수 정의</h5>
    <h5>4. sound : 게임 진행 음성 파일 폴더</h5>
    <h5>5. mugunghwa.py : 무궁화 게임 메인 서버</h5>
</p>


## System Structure
<p align="center">
    <img src="https://user-images.githubusercontent.com/78125184/148163269-492f7c99-41c2-43ef-8170-5182d8730ff2.png"/>
</p>


## Contributor

Maintainer : 김수연, 김서정

Contributor : 김남협, 김주환, 박수정, 유동헌, 윤혜정, 조민호, 전민준, 허나연



## License

MIT License
