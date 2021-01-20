import cv2,numpy as np,math
hand_hist = None

def cap_rect(frame):
	#row,cols,_=frame.shape()
	cv2.rectangle(frame,(200,200),(400,400),(0,255,0),0)
	return frame

def calc_hist(frame):
	hsv_crop=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	roi = np.zeros([10, 10, 3], dtype=hsv_crop.dtype)
	roi=hsv_crop[200:400,200:400]
	cal_hist=cv2.calcHist([roi],[0,1],None,[180,255],[0,180,0,255])
	#cv2.imshow('cal_hist',cal_hist)
	d=cv2.normalize(cal_hist,cal_hist,0,255,cv2.NORM_MINMAX)
	#cv2.imshow('d',d)
	return cv2.normalize(cal_hist,cal_hist,0,255,cv2.NORM_MINMAX)

def hist_masking(frame,hist):
	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	mask1=cv2.calcBackProject([hsv],[0,1],hand_hist,[0,180,0,256],1)
	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
	cv2.filter2D(mask1,-1,disc,mask1)
	_,thresh1=cv2.threshold(mask1,150,255,cv2.THRESH_BINARY)
	thresh1=cv2.merge((thresh1,thresh1,thresh1))
	cv2.imshow('thresh',thresh1)
	return cv2.bitwise_and(frame,thresh1)

def main():
	global hand_hist
	hist_created=False
	cap=cv2.VideoCapture(0)
	while cap.isOpened():
		color_key=cv2.waitKey(1)
		ret, frame = cap.read()
		frame=cv2.flip(frame,1)
		img=frame.copy()
		if color_key & 0xFF == ord('z'):
			hist_created=True
			hand_hist= calc_hist(frame)
		if hist_created:
			hist_mask_img=hist_masking(frame,hand_hist)
			hist_mask_img= cv2.erode(hist_mask_img, None, iterations=2)
			hist_mask_img = cv2.dilate(hist_mask_img, None, iterations=2)


			#cv2.imshow('img',frame)

			
			gray_hist_mask=cv2.cvtColor(hist_mask_img,cv2.COLOR_BGR2GRAY)
			_,thresh2=cv2.threshold(gray_hist_mask,0,255,0)
			cnts,hie=cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			cnt = max(cnts, key=cv2.contourArea)

			x,y,w,h = cv2.boundingRect(cnt)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),0)
			hull = cv2.convexHull(cnt)
			drawing = np.zeros(frame.shape,np.uint8)
			cv2.drawContours(drawing,[cnt],0,(0,255,0),0)
			cv2.drawContours(drawing,[hull],0,(0,0,255),0)
			hull = cv2.convexHull(cnt,returnPoints = False)
			defects = cv2.convexityDefects(cnt,hull)
			count_defects = 0
			cv2.drawContours(thresh2, cnts, -1, (0,255,0), 3)
			for i in range(defects.shape[0]):
				s,e,f,d = defects[i,0]
				start = tuple(cnt[s][0])
				end = tuple(cnt[e][0])
				far = tuple(cnt[f][0])
				a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
				b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
				c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
				angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
				if angle <= 90:
					count_defects += 1
					cv2.circle(frame,far,1,[0,0,255],-1)
				cv2.line(frame,start,end,[0,255,0],2)
			if count_defects == 1:
				cv2.putText(frame, "TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
			elif count_defects == 2:
				cv2.putText(frame, "THREE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
			elif count_defects == 3:
				cv2.putText(frame, "FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
			elif count_defects == 4:
				cv2.putText(frame, "FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
			elif count_defects == 0:
				cv2.putText(frame, "ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
			else:
				pass

				
		else:
			frame=cap_rect(frame)

			#res,thresh=np.vstack((thresh,res))
			#cv2.imshow('res',res)
		cv2.imshow('frame',frame)
		if cv2.waitKey(30)==ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
if __name__ == '__main__':
	main()
