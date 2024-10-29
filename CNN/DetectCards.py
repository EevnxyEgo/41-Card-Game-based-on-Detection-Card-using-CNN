import sys
sys.path.insert(0,"..")
from game.Game import Player, Computer

import cv2
import numpy as np
from keras.models import load_model
from collections import defaultdict

class CardDetector:
    def __init__(self, model_path, folder_path, width=640, height=480):
        self.WIDTH_IMAGE = width
        self.state = None
        self.HEIGHT_IMAGE = height
        self.cap = cv2.VideoCapture(1)
        self.cap.set(10, 130)
        self.model = load_model(model_path)
        self.FOLDER_PATH = folder_path
        self.CLASSES = ['ace of spades', 'two of spades', 'three of spades', 'four of spades', 'five of spades',
                   'six of spades', 'seven of spades', 'eight of spades', 'nine of spades', 'ten of spades',
                   'jack of spades', 'queen of spades', 'king of spades', 'ace of clubs', 'two of clubs', 
                   'three of clubs', 'four of clubs', 'five of clubs', 'six of clubs', 'seven of clubs', 
                   'eight of clubs', 'nine of clubs', 'ten of clubs', 'jack of clubs', 'queen of clubs', 
                   'king of clubs', 'ace of diamonds', 'two of diamonds', 'three of diamonds', 
                   'four of diamonds', 'five of diamonds', 'six of diamonds', 'seven of diamonds', 
                   'eight of diamonds', 'nine of diamonds','ten of diamonds', 'jack of diamonds', 
                   'queen of diamonds', 'king of diamonds', 'ace of hearts', 'two of hearts', 
                   'three of hearts', 'four of hearts', 'five of hearts', 'six of hearts', 'seven of hearts', 
                   'eight of hearts', 'nine of hearts', 'ten of hearts', 'jack of hearts', 'queen of hearts', 
                   'king of hearts']
        # print(self.CLASSES)
        self.player1 = Player()
        self.computer1 = Computer()
        self.display_image = cv2.imread('C:/Users/arsen/PCV/Big Project/game/Background.png')

    def predict_image(self, image):
        predictions = self.model.predict(np.array([image]))
        class_index = int(np.argmax(predictions))
        class_name = self.CLASSES[class_index]
        confidence = predictions[0, class_index]
        return confidence, class_name, class_index

    def preprocess_image(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
        img_canny = cv2.Canny(img_blur, 200, 200)
        kernel = np.ones((5, 5))
        img_dilated = cv2.dilate(img_canny, kernel, iterations=2)
        img_thresh = cv2.erode(img_dilated, kernel, iterations=1)
        return img_thresh

    def find_biggest_contour(self, image):
        biggest_contour = np.array([])
        max_area = 0
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:
                arc_length = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * arc_length, True)
                if area > max_area and len(approx) == 4:
                    biggest_contour = approx
                    max_area = area
        return biggest_contour

    def rearrange_corner_points(self, corner_points):
        corner_points = corner_points.reshape((4, 2))
        corner_points_new = np.zeros((4, 1, 2), np.int32)
        total = corner_points.sum(1)

        corner_points_new[0] = corner_points[np.argmin(total)]
        corner_points_new[3] = corner_points[np.argmax(total)]

        diff = np.diff(corner_points, axis=1)
        corner_points_new[1] = corner_points[np.argmin(diff)]
        corner_points_new[2] = corner_points[np.argmax(diff)]
        return corner_points_new

    def warp_perspective(self, image, biggest_contour):
        biggest = self.rearrange_corner_points(biggest_contour)
        matrix_warp1 = np.float32([biggest[0], biggest[1], biggest[2], biggest[3]])
        matrix_warp2 = np.float32([[0, 0], [self.WIDTH_IMAGE, 0], [0, self.HEIGHT_IMAGE], [self.WIDTH_IMAGE, self.HEIGHT_IMAGE]])
        matrix = cv2.getPerspectiveTransform(matrix_warp1, matrix_warp2)
        warped_image = cv2.warpPerspective(image, matrix, (self.WIDTH_IMAGE, self.HEIGHT_IMAGE))
        return warped_image
    
    def show_card_images(self, deck, position):
        x_offset = 0
        for card_value in deck:
            card_image = cv2.imread(f"C:/Users/arsen/PCV/Big Project/game/assetCards/{card_value}.png")  
            card_image = cv2.resize(card_image, (56, 84))  
            
            if self.display_image is None:
                self.display_image = np.zeros((480, 640, 3), np.uint8)
    
            x, y = position
            y_offset = 200 if y == self.HEIGHT_IMAGE else 0
    
            self.display_image[y+y_offset:y+84+y_offset, x+x_offset:x+56+x_offset] = card_image  
            x_offset += 91
                      
    
    def take_out_card(self, current_player):
        playerSuit = np.array([kartu[1] for kartu in current_player.PlayerCard])
        
        suitValue, counts = np.unique(playerSuit, return_counts=True)
        leastCommonSuit = suitValue[np.argmin(counts)]
        mostCommonSuit = suitValue[np.argmax(counts)]
        print("most common",mostCommonSuit)
        
        if leastCommonSuit == mostCommonSuit and counts[np.argmax(counts)] == 5:
            lowestValue = float('inf')
            lowestSuitCard = None
            for card in current_player.PlayerCard:
                if card[1] == leastCommonSuit and card[0] < lowestValue:
                    lowestValue = card[0]
                    lowestSuitCard = card
            
            if lowestSuitCard is not None:
                x = current_player.PlayerCard.index(lowestSuitCard)
                print("index yang dikeluarkan:", x)
                print("sebelum Pop")
                print(current_player.deck)
                print(current_player.PlayerCard)
                trash_value = current_player.deck.pop(x)
                print(f"Kartu yang dikeluarkan adalah {lowestSuitCard}")
                trash_card = current_player.PlayerCard.pop(x)
                print(current_player.deck)
                print(current_player.PlayerCard)
                current_player.already_shown.remove(trash_value)
                return trash_card, trash_value
            else:
                print("Tidak ada kartu dengan kondisi yang diberikan.")
        
        elif np.max(counts)==2:
            
            suits = np.array([card[1] for card in current_player.PlayerCard])
            unique_suits, suit_counts = np.unique(suits, return_counts=True)
            # print(unique_suits)
            max_count = np.max(suit_counts)
            result = [(value, suit) for value, suit in current_player.PlayerCard if np.sum(suits == suit) == max_count]
            
            suit_values = defaultdict(list)
            
            for value, suit in result:
                suit_values[suit].append(value)
            # print(suit_values)
            if len(suit_values[suit])==4:
                total_values = {suit: sum(suit_values[suit]) for suit in suit_values}
                
                min_values = min(total_values.values())
                
                # Membuat list baru dengan elemen yang memiliki jumlah nilai paling sedikit untuk setiap suit
                Highbutlow = [(value, suit) for suit, val in total_values.items() if val == min_values for value in suit_values[suit]]
                
                lowest = min((value, suit) for value, suit in current_player.PlayerCard if suit == leastCommonSuit)
                
                Highbutlow.append(lowest)
                
                highestValue = max([card[0] for card in Highbutlow])
    
                lowestSuitCard = None
                for card in Highbutlow:
                     if card[0] == highestValue:
                         lowestSuitCard = card
                         break
            else:
                lowestsuits = np.array([card[1] for card in current_player.PlayerCard])
                unique_suits, suit_counts = np.unique(lowestsuits, return_counts=True)
                print(unique_suits)
                min_count = np.min(suit_counts)

                result = [(value, suit) for value, suit in current_player.PlayerCard if np.sum(lowestsuits == suit) == min_count]

                highestValue = max([card[0] for card in result])

                lowestSuitCard = None
                for card in result:
                     if card[0] == highestValue:
                         lowestSuitCard = card
                         break

            if lowestSuitCard is not None:
                x = current_player.PlayerCard.index(lowestSuitCard)
                print("index yang dikeluarkan:", x)
                print("sebelum Pop")
                print(current_player.deck)
                print(current_player.PlayerCard)
                trash_value = current_player.deck.pop(x)
                print(f"Kartu yang dikeluarkan adalah {lowestSuitCard}")
                trash_card = current_player.PlayerCard.pop(x)
                print(current_player.deck)
                print(current_player.PlayerCard)
                current_player.already_shown.remove(trash_value)
                return trash_card, trash_value
            else:
                print("Tidak ada kartu dengan suit yang paling sedikit dan nilai tertinggi.")
                
        else:
            lowestsuits = np.array([card[1] for card in current_player.PlayerCard])
            unique_suits, suit_counts = np.unique(lowestsuits, return_counts=True)
            print(unique_suits)
            min_count = np.min(suit_counts)

            result = [(value, suit) for value, suit in current_player.PlayerCard if np.sum(lowestsuits == suit) == min_count]

            highestValue = max([card[0] for card in result])

            lowestSuitCard = None
            for card in result:
                 if card[0] == highestValue:
                     lowestSuitCard = card
                     break

            if lowestSuitCard is not None:
                x = current_player.PlayerCard.index(lowestSuitCard)
                print("index yang dikeluarkan:", x)
                print("sebelum Pop")
                print(current_player.deck)
                print(current_player.PlayerCard)
                trash_value = current_player.deck.pop(x)
                print(f"Kartu yang dikeluarkan adalah {lowestSuitCard}")
                trash_card = current_player.PlayerCard.pop(x)
                print(current_player.deck)
                print(current_player.PlayerCard)
                current_player.already_shown.remove(trash_value)
                return trash_card, trash_value
            else:
                print("Tidak ada kartu dengan suit yang paling sedikit dan nilai tertinggi.")



    def take_turn(self, player, class_index):
        if player == "player":
            current_player = self.player1
            other_player = self.computer1
        else:
            current_player = self.computer1
            other_player = self.player1
        
        if current_player.IsDeckFull():
            # Player mengambil satu kartu lagi
            if (len(other_player.trashForm) >= 1 and len(other_player.trash) >= 1):
                suitTrash = np.array([suit[1] for suit in other_player.trashForm])
                playerSuit = np.array([kartu[1] for kartu in current_player.PlayerCard])
                suitValue, counts = np.unique(playerSuit, return_counts=True)
                mostCommonSuit = suitValue[np.argmax(counts)]
                
                if np.array_equal(suitTrash, mostCommonSuit):
                    s = [str(i) for i in other_player.trash]
                    x = int("".join(s))
                    new_card_index = x-1
                    other_player.trash.pop(0)
                    other_player.trashForm.pop(0)
                else:
                    new_card_index = class_index
            else:
                new_card_index = class_index
                
            if new_card_index+1 not in other_player.already_shown and new_card_index+1 not in current_player.already_shown:
                print(current_player.deck)
                #print(f"{player.capitalize()} mengambil kartu:{class_index}")
                if cv2.waitKey(1) & 0xFF == ord('y'):
                    current_player.AppendCard(new_card_index)
                    card_to_trash, trash_value = self.take_out_card(current_player)
                    print(f"{player.capitalize()} membuang kartu: {card_to_trash}")
                    current_player.trash.append(trash_value)
                    current_player.trashForm.append(card_to_trash)
                    
                    if(len(current_player.trash) == 2 and len(current_player.trashForm)==2):
                        current_player.trashForm.pop(0)
                        current_player.trash.pop(0)
                        print(current_player.trash)

            # Menampilkan deck setelah mengambil dan membuang kartu
        self.show_card_images(current_player.deck, (155, 296))
        self.show_card_images(other_player.deck, (155, 100))        
    
    def restart_game(self):
        self.player1.Reset()  # Mengosongkan kartu dan skor pemain
        self.computer1.Reset()
        self.show_card_images(self.player1.deck, (155, 296))
        self.show_card_images(self.computer1.deck, (155, 100)) 
        self.show_card_images(self.player1.trash, (50, 198))
        self.show_card_images(self.computer1.trash, (50, 198))# Mengosongkan kartu dan skor komputer
        if cv2.waitKey(1) & 0xFF == ord('r'):
            self.run_detection()
        
        
    def run_detection(self):
        
        player_turn = "player"
        while True:
            success, img = self.cap.read()

            if not success:
                break

            img = cv2.resize(img, (self.WIDTH_IMAGE, self.HEIGHT_IMAGE))
            # img_contour = img.copy()

            img_thresh = self.preprocess_image(img)
            biggest_contour = self.find_biggest_contour(img_thresh)

            if biggest_contour.size != 0:
                img_warped = self.warp_perspective(img, biggest_contour)
                img_warped_resized = cv2.resize(img_warped, (200, 300))
                img_warped_gray = cv2.cvtColor(img_warped_resized, cv2.COLOR_BGR2GRAY)
                img_thresh1 = cv2.adaptiveThreshold(img_warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 67, 11)
                normalized_image = img_thresh1 / 255.0
                normalized_image = cv2.resize(normalized_image, (128, 128))
                cv2.imshow("warped", img_thresh1)
                confidence, class_name, class_index = self.predict_image(normalized_image)
                
                if confidence > 0.85:
                    cv2.putText(img, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    cv2.rectangle(self.display_image, (84, 418), (210, 390), (32, 78, 6), -1)
                    
                    
                    if self.player1.IsDeckFull():
                        if class_index+1 not in self.player1.deck and class_index+1 not in self.computer1.deck:
                            if not self.computer1.IsDeckFull():
                                cv2.putText(self.display_image, "COM'S TURN", (426, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                
                                if cv2.waitKey(1) & 0xFF == ord('y'):
                                    self.computer1.AppendCard(class_index)
                                    computer_score = self.computer1.CalculateScore()
                                    cv2.rectangle(self.display_image, (263, 44), (376, 81), (0, 0, 0), -1)
                                    cv2.putText(self.display_image, f"Score: {computer_score}", (278, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                    elif not self.player1.IsDeckFull():
                        
                        cv2.putText(self.display_image, "PLAYER'S TURN", (86, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        if cv2.waitKey(1) & 0xFF == ord('y'):
                            self.player1.AppendCard(class_index)
                            player_score=self.player1.CalculateScore()
                            cv2.rectangle(self.display_image, (263, 400), (376, 436), (0, 0, 0), -1)
                            cv2.putText(self.display_image, f"Score : {player_score}", (278, 426), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                if self.player1.IsDeckFull() and self.computer1.IsDeckFull():
                    cv2.rectangle(self.display_image, (84, 418), (210, 390), (32, 78, 6), -1)
                    cv2.rectangle(self.display_image, (421, 44), (550, 80), (32, 78, 6), -1)
                    if player_turn == "player":
                        print("giliran player")
                        cv2.putText(self.display_image, "PLAYER'S TURN", (86, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        self.take_turn("player",class_index)
                        self.show_card_images(self.player1.trash, (532, 198))
                        player_score=self.player1.CalculateScore()
                        cv2.rectangle(self.display_image, (263, 400), (376, 436), (0, 0, 0), -1)
                        cv2.putText(self.display_image, f"Score : {player_score}", (278, 426), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        if self.player1.CalculateScore() == 41:
                            cv2.putText(self.display_image, "YOU WIN", (250, 235), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            self.restart_game()
                        if cv2.waitKey(1) & 0xFF == ord('c'):
                            player_turn = "computer"
                    else:   
                        print("giliran com")
                        cv2.putText(self.display_image, "COM'S TURN", (426, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        self.take_turn("computer",class_index)
                        self.show_card_images(self.computer1.trash, (50, 198))
                        computer_score = self.computer1.CalculateScore()
                        cv2.rectangle(self.display_image, (263, 44), (376, 81), (0, 0, 0), -1)
                        cv2.putText(self.display_image, f"Score: {computer_score}", (278, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        if self.computer1.CalculateScore() == 41:
                            cv2.putText(self.display_image, "NICE TRY", (250, 235), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            self.restart_game()
                        if cv2.waitKey(1) & 0xFF == ord('c'):
                            player_turn = "player"
            

            else:
                cv2.putText(img, 'No Card Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            vertical_appended_img = np.vstack((img_thresh))
            cv2.imshow("Contour", vertical_appended_img)
            cv2.imshow("Result", img)
            
        

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            self.show_card_images(self.player1.deck, (155, 296))
        
            self.show_card_images(self.computer1.deck, (155, 100))

            cv2.imshow("Display", self.display_image)

        self.cap.release()
        cv2.destroyAllWindows()
        

#run game
model_path = 'C:/Users/arsen/PCV/Big Project/CNN/DatasetH5PY/ALLINONE.h5'
folder_path = r"C:/Users/arsen/PCV/Big Project/CNN/Dataset/Training/Images"

card_detector = CardDetector(model_path, folder_path)
card_detector.run_detection()
