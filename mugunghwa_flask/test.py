import time

in_sec = input("시간을 입력하세요.(초):")
sec = int(in_sec)
print(sec)

#while은 반복문으로 sec가 0이 되면 반복을 멈춰라
while (sec != 0 ):
    sec = sec-1
    time.sleep(1)
    print(sec)