import random
import numpy as np
from collections import defaultdict

def CardValue(CardNumber):
    CardNumber = (CardNumber - 1) % 13 + 1 
    if CardNumber >= 2 and CardNumber <= 9:#2 to 9
        return CardNumber
    elif CardNumber in [10, 11, 12, 13]:#10, J, Q, K
        return 10
    elif CardNumber == 1:#As
        return 11

def SuitValue(CardNumber):
    if CardNumber <= 13:
        return 1  # Sekop
    elif CardNumber <= 26:
        return 2  # Keriting
    elif CardNumber <= 39:
        return 3  # wajik
    else:
        return 4  # hati

def Transform():
    PlayerCard = []
    #nomor_sudah = []
    Playerdeck = [1,39,18,43,44]
    
    for i in range(len(Playerdeck)):
        # CardNumber = random.randint(1, 52)
        
        # while CardNumber in nomor_sudah:
        #     CardNumber = random.randint(1, 52)  
        
        # nomor_sudah.append(CardNumber)
        # Playerdeck.append(CardNumber)
        CardNumber = Playerdeck[i]
        cardValue = CardValue(CardNumber)
        suitValue = SuitValue(CardNumber)
        PlayerCard.append((cardValue, suitValue)) 
    return PlayerCard, Playerdeck

def TakeOutCard(PlayerCard, Playerdeck):
    playerSuit = np.array([kartu[1] for kartu in PlayerCard])

    suitValue, counts = np.unique(playerSuit, return_counts=True)
    
    print(suitValue,counts)
    
    leastCommonSuit = suitValue[np.argmin(counts)]
    mostCommonSuit = suitValue[np.argmax(counts)]
    # print("most common",mostCommonSuit)

    if leastCommonSuit == mostCommonSuit and counts[np.argmax(counts)] == 5:
        lowestValue = float('inf')
        lowestSuitCard = None
        for card in PlayerCard:
            if card[1] == leastCommonSuit and card[0] < lowestValue:
                lowestValue = card[0]
                lowestSuitCard = card
        
        if lowestSuitCard is not None:
            x = PlayerCard.index(lowestSuitCard)
            print("index yang dikeluarkan:", x)
            print("sebelum Pop")
            print(Playerdeck)
            print(PlayerCard)
            Playerdeck.pop(x)
            print(f"Kartu yang dikeluarkan adalah {lowestSuitCard}")
            PlayerCard.pop(x)
            print(Playerdeck)
            print(PlayerCard)
        else:
            print("Tidak ada kartu dengan kondisi yang diberikan.")
            
    elif np.max(counts)==2:
        
        suits = np.array([card[1] for card in PlayerCard])
        unique_suits, suit_counts = np.unique(suits, return_counts=True)
        # print(unique_suits)
        max_count = np.max(suit_counts)
        result = [(value, suit) for value, suit in PlayerCard if np.sum(suits == suit) == max_count]
        
        suit_values = defaultdict(list)
        
        for value, suit in result:
            suit_values[suit].append(value)
        print("anjay",len(suit_values[suit]))
        
        total_values = {suit: sum(suit_values[suit]) for suit in suit_values}
        
        min_values = min(total_values.values())
        
        # Membuat list baru dengan elemen yang memiliki jumlah nilai paling sedikit untuk setiap suit
        Highbutlow = [(value, suit) for suit, val in total_values.items() if val == min_values for value in suit_values[suit]]
        
        lowest = min((value, suit) for value, suit in PlayerCard if suit == leastCommonSuit)
        
        Highbutlow.append(lowest)
        
        highestValue = max([card[0] for card in Highbutlow])

        lowestSuitCard = None
        for card in Highbutlow:
             if card[0] == highestValue:
                 lowestSuitCard = card
                 break

        if lowestSuitCard is not None:
            x = PlayerCard.index(lowestSuitCard)
            print("index yang dikeluarkan:", x)
            print("sebelum Pop")
            print(Playerdeck)
            print(PlayerCard)
            Playerdeck.pop(x)
            print(f"Kartu yang dikeluarkan adalah {lowestSuitCard}")
            PlayerCard.pop(x)
            print(Playerdeck)
            print(PlayerCard)
        else:
            print("Tidak ada kartu dengan suit yang paling sedikit dan nilai tertinggi.")
        
        print(Highbutlow)

        
    else:
        
        lowestsuits = np.array([card[1] for card in PlayerCard])
        unique_suits, suit_counts = np.unique(lowestsuits, return_counts=True)
        # print(unique_suits)
        min_count = np.min(suit_counts)
        
        result = [(value, suit) for value, suit in PlayerCard if np.sum(lowestsuits == suit) == min_count]
        
        highestValue = max([card[0] for card in result])

        lowestSuitCard = None
        for card in result:
             if card[0] == highestValue:
                 lowestSuitCard = card
                 break

        if lowestSuitCard is not None:
            x = PlayerCard.index(lowestSuitCard)
            print("index yang dikeluarkan:", x)
            print("sebelum Pop")
            print(Playerdeck)
            print(PlayerCard)
            Playerdeck.pop(x)
            print(f"Kartu yang dikeluarkan adalah {lowestSuitCard}")
            PlayerCard.pop(x)
            print(Playerdeck)
            print(PlayerCard)
        else:
            print("Tidak ada kartu dengan suit yang paling sedikit dan nilai tertinggi.")






    

def CalculateScore(PlayerCard):
        suit_values = defaultdict(list)
        
        for value, suit in PlayerCard:
            suit_values[suit].append(value)
        
        total_values = {suit: sum(suit_values[suit]) for suit in suit_values}
        
        max_values = max(total_values.values())
        score = max_values
        result = [(value, suit) for suit, val in total_values.items() if val == max_values for value in suit_values[suit]]
        
        
        for kartu in PlayerCard:
            if kartu[1] != result[0][1]:
                score -= kartu[0]
                
        return score
    
PlayerCard, Playerdeck = Transform()
TakeOutCard(PlayerCard, Playerdeck)
PlayerScore = CalculateScore(PlayerCard)
print(f"Kartu Pemain: {PlayerCard}") 
print(f"Deck Pemain: {Playerdeck}")
print(f"Skor Pemain: {PlayerScore}")
