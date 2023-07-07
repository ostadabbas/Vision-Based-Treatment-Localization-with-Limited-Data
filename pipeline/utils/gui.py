import time
import cv2
import numpy as np

class GUI:

    outputFrame = None
    tccc_template = cv2.imread('utils/media/tccc.png')
    tccc_img = tccc_template.copy()
    console_img = np.zeros((360,1280,3), np.uint8)
    prev_frame_time=0

    def __init__(self) -> None:
        pass

    def drawGUI(self):

        '''
        Combine all the images into one GUI interface

        PARAMETERS:
            None

        OUTPUT:
            gui: GUI image (np.array)
        '''

        # Resize output frame to 1280x720
        if self.outputFrame.shape[1]!= 1280:
                self.outputFrame = cv2.resize(self.outputFrame,(1280,720), interpolation = cv2.INTER_AREA)

        # Concatenate all images
        gui = np.concatenate((self.outputFrame, self.console_img), axis=0)
        gui = np.concatenate((gui, self.tccc_img), axis=1)

        # Display fps
        self.new_frame_time = time.time()
        fps = f"FPS: {int(1/(self.new_frame_time-self.prev_frame_time))}"
        elapsed = f"Elapsed: {int((self.new_frame_time-self.prev_frame_time)*1000)}ms"
        cv2.putText(gui, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(gui, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(gui, elapsed, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(gui, elapsed, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
        self.prev_frame_time = self.new_frame_time

        return cv2.resize(gui,(1920,1080))

    def updateConsole(self, text):

        '''
        Updates the console with the text

        PARAMETERS:
            Text: Text to be displayed on the console (String)

        OUTPUT:
            None
        '''

        console_template = np.zeros((60,1280,3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        textY = int((console_template.shape[0] + textsize[1]) / 2)
        cv2.putText(console_template, text, (10, textY), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        self.console_img = np.concatenate((self.console_img, console_template), axis=0)
        self.console_img = self.console_img[60:420, 0:1280]

    def updateOutputFrame(self, frame):

        '''
        Updates the output frame

        PARAMETERS:
            frame: Output frame (np.array)

        OUTPUT:
            None
        '''
        self.outputFrame = frame

    def markTCCCLimb(self, limb, treatment, radius, color):

        circle_loc = None
        match limb:
            case 'right_arm':
                circle_loc = (234, 470)
                text_loc = (133, 419)
            case 'left_arm':
                circle_loc = (333, 470)
                text_loc = (416, 419)
            case 'right_leg':
                circle_loc = (260, 650)
                text_loc = (133, 684)
            case 'left_leg':
                circle_loc = (307, 650)
                text_loc = (411, 684)

        if circle_loc:
            self.tccc_img = cv2.circle(
                self.tccc_img, circle_loc, int(radius+0.5), color, -1)
            cv2.putText(self.tccc_img, f"{treatment}", text_loc,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3, cv2.LINE_AA)
            cv2.putText(self.tccc_img, f"{treatment}", text_loc,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 0, 0], 1, cv2.LINE_AA)


#    def markTCCCLimb(self, limb, treatment, radius, color):
#
#        match limb:
#            case 'right_arm':
#                self.tccc_img = cv2.circle(
#                    self.tccc_img, (234, 470), int(radius+0.5), color, -1)
#                cv2.putText(self.tccc_img, f"{treatment}", (133, 419),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
#            case 'left_arm':
#                self.tccc_img = cv2.circle(
#                    self.tccc_img, (333, 470), int(radius+0.5), color, -1)
#                cv2.putText(self.tccc_img, f"{treatment}", (416, 419),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
#            case 'right_leg':
#                self.tccc_img = cv2.circle(
#                    self.tccc_img, (260, 650), int(radius+0.5), color, -1)
#                cv2.putText(self.tccc_img, f"{treatment}", (133,684),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
#            case 'left_leg':
#                self.tccc_img = cv2.circle(
#                    self.tccc_img, (307, 650), int(radius+0.5), color, -1)
#                cv2.putText(self.tccc_img, f"{treatment}", (411,684),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)


    # TODO: Incomplete
    def markTCCC(self, joint):

        '''
        Draws a circle on the TCCC image to mark the location of the intervention

        PARAMETERS:
            cls_name: Class name of the intervention (String)
            tracker_id: Tracker ID of the intervention (Int)
            joint:  keypoint min_idx (Int)


        OUTPUT:
            None
        '''

        match joint: # TODO: Assuming Back is different, assuming all wounds are front side rn.. no exit wounds :)
            case 2: # r_sho
                self.tccc_img = cv2.circle(self.tccc_img, (237,440), 8, [0,0,255], -1)
            case 3: # r_elb
                self.tccc_img = cv2.circle(self.tccc_img, (230,500), 8, [0,0,255], -1)
            case 4: # r_wri
                self.tccc_img = cv2.circle(self.tccc_img, (210,550), 8, [0,0,255], -1)
            case 5: # l_sho
                self.tccc_img = cv2.circle(self.tccc_img, (330,440), 8, [0,0,255], -1)
            case 6: # l_elb
                self.tccc_img = cv2.circle(self.tccc_img, (335,500), 8, [0,0,255], -1)
            case 7: # l_wri
                self.tccc_img = cv2.circle(self.tccc_img, (355,550), 8, [0,0,255], -1)
            case 8: # r_hip
                self.tccc_img = cv2.circle(self.tccc_img, (262,550), 8, [0,0,255], -1)
            case 9: # r_knee
                self.tccc_img = cv2.circle(self.tccc_img, (260,650), 8, [0,0,255], -1)
            case 10: # r_ank
                self.tccc_img = cv2.circle(self.tccc_img, (265,730), 8, [0,0,255], -1)
            case 11: # l_hip
                self.tccc_img = cv2.circle(self.tccc_img, (305,550), 8, [0,0,255], -1)
            case 12: # l_knee
                self.tccc_img = cv2.circle(self.tccc_img, (307,650), 8, [0,0,255], -1)
            case 13: # l_ank
                self.tccc_img = cv2.circle(self.tccc_img, (302,862), 8, [0,0,255], -1)
            case _:
                if joint is not None: # head
                    self.tccc_img = cv2.circle(self.tccc_img, (283,380), 8, [0,0,255], -1)

    def clearTCCC(self):
        '''
        Clears the TCCC image

        PARAMETERS:
            None

        OUTPUT:
            None

        '''
        self.tccc_img = self.tccc_template.copy()
