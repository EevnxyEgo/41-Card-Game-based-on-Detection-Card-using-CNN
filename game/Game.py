import sys
sys.path.insert(0,"..") 
import numpy as np
from collections import defaultdict

class Player:
    def __init__(self):
        self.deck = []
        self.score = None
        print(f"player created")
        self.already_shown = [] 
        self.PlayerCard = []
        self.PlayerCard_shown =[]
        self.trash = []
        self.trashForm = []

        
    def AppendCard(self,cardvalue):
        if cardvalue+1 not in self.already_shown:
            self.already_shown.append(cardvalue+1)
            self.deck.append(cardvalue+1)
            cardValue = self.CardValue(cardvalue+1)
            suitValue = self.SuitValue(cardvalue+1)
            self.PlayerCard.append((cardValue, suitValue))
            print("kartu Player:",self.deck)
            print("dek form Player:",self.PlayerCard)

    def IsDeckFull(self):
        return len(self.deck) == 4
    
    def CardValue(self,cardvalue):
        cardvalue = (cardvalue - 1) % 13 + 1 
        if cardvalue >= 2 and cardvalue <= 9:
            return cardvalue
        elif cardvalue in [10, 11, 12, 13]:
            return 10
        elif cardvalue == 1:
            return 11

    def SuitValue(self, cardvalue):
        if cardvalue <= 13:
            return 1  # Sekop
        elif cardvalue <= 26:
            return 2  # Keriting
        elif cardvalue <= 39:
            return 3  # Wajik
        else:
            return 4  # Hati

    def CalculateScore(self):
        
        suit_values = defaultdict(list)

        for value, suit in self.PlayerCard:
            suit_values[suit].append(value)
        
        total_values = {suit: sum(suit_values[suit]) for suit in suit_values}
        
        max_values = max(total_values.values())
        self.score = max_values
        result = [(value, suit) for suit, val in total_values.items() if val == max_values for value in suit_values[suit]]
        
        for kartu in self.PlayerCard:
            if kartu[1] != result[0][1]:
                self.score -= kartu[0]
                
        return self.score
    
    def Reset(self):
        self.deck = []
        self.score = None
        self.already_shown = [] 
        self.PlayerCard = []
        self.PlayerCard_shown =[]
        self.trash = []
        self.trashForm = []
    

    
class Computer:
    def __init__(self):
        self.deck = []
        self.score = None
        print(f"Computer created")
        self.already_shown = []
        self.PlayerCard = []
        self.ComputerCard_shown = []
        self.trash = []
        self.trashForm = []
        
    def AppendCard(self,cardvalue):
        if cardvalue+1 not in self.already_shown:
            self.already_shown.append(cardvalue+1)
            self.deck.append(cardvalue+1)
            cardValue = self.CardValue(cardvalue+1)
            suitValue = self.SuitValue(cardvalue+1)
            self.PlayerCard.append((cardValue, suitValue))
            print("Kartu Com: ",self.deck)
            print("dek form Com:",self.PlayerCard)
            
    def IsDeckFull(self):
        return len(self.deck) == 4
    
    def CardValue(self,cardvalue):
        cardvalue = (cardvalue - 1) % 13 + 1 
        if cardvalue >= 2 and cardvalue <= 9:
            return cardvalue
        elif cardvalue in [10, 11, 12, 13]:
            return 10
        elif cardvalue == 1:
            return 11

    def SuitValue(self, cardvalue):
        if cardvalue <= 13:
            return 1  # Sekop
        elif cardvalue <= 26:
            return 2  # Keriting
        elif cardvalue <= 39:
            return 3  # Wajik
        else:
            return 4  # Hati

    def CalculateScore(self):   
        suit_values = defaultdict(list)

        for value, suit in self.PlayerCard:
            suit_values[suit].append(value)
        
        total_values = {suit: sum(suit_values[suit]) for suit in suit_values}
        
        max_values = max(total_values.values())
        self.score = max_values
        result = [(value, suit) for suit, val in total_values.items() if val == max_values for value in suit_values[suit]]
        
        for kartu in self.PlayerCard:
            if kartu[1] != result[0][1]:
                self.score -= kartu[0]
                
        return self.score
    
    def Reset(self):
        self.deck = []
        self.score = None
        self.already_shown = [] 
        self.PlayerCard = []
        self.ComputerCard_shown =[]
        self.trash = []
        self.trashForm = []
        

