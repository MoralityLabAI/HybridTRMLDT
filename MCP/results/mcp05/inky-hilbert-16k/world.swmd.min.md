---
title: "The City That Lost Its Night Bell"
storyworld_id: SW-INKY-HILBERT-16384-20261124
source_format: SWMD-0-MIN
hilbert_player_data: storyworld.hilbert.json
state_variables: ["real_00", "real_01", "real_10", "real_11", "imag_00", "imag_01", "imag_10", "imag_11", "c0", "c1", "c2", "c3", "c4", "c5"]
endings:
  - id: end_bell_returns
    type: public
    condition: "The bell rings again, restored to the city, and the dead are called home"
    clue_chain: "Cracked bell + clapper missing + Keeper's ledger + mechanism as key"
  - id: end_bell_sold
    type: public
    condition: "Corvin sells the bell's sound to a foreign power, and the city wakes to a stranger's hour"
    clue_chain: "Ledger of hours + clapper missing + mechanism as key"
  - id: end_bell_lost_forever
    type: public
    condition: "The bell is lost to the sea, and the city learns to keep its own time"
    clue_chain: "Cracked bell + Keeper's ledger + mechanism as key"
  - id: end_bell_as_question
    type: secret
    condition: "The bell is rung as a question, and the city must answer it together"
    clue_chain: "Cracked bell + Keeper's ledger + mechanism as key + Pip's questions"
  - id: end_bell_as_answer
    type: secret
    condition: "The bell is rung as an answer, and the city must decide what it was hiding"
    clue_chain: "Cracked bell + clapper missing + Keeper's ledger + Pip's questions"
---

# SWMD-0-MIN
id: SW-INKY-HILBERT-16384-20261124
title: The City That Lost Its Night Bell

ENC enc_01 turn=0..999
# TITLE: The Silence at Dusk
# T: The bell does not ring for the hour; it rings for the dead, and the dead do not care what the merchants call it.
# T: 
# T: "They say the old bell marked the end of the day," said Maren, standing where the stair meets the square. "But I have heard the ledger of hours open on the merchant's table, and every coin and every bargain is recorded in the name of the old order."
# T: 
# T: "And the keeper's cellar is dark," said Corvin, turning a coin over in his fingers. "The old mechanism waits in the dark for someone to understand what it was built to do."
# T: 
# T: "If the bell is cracked, then something inside it is cracked too," said Pip, crouching by the broken clapper. "And if the clapper is missing, then the bell is missing its voice."
# T: 
# T: "The bell is sold to a stranger and the city wakes to a foreign hour," said Maren. "But the city holds its breath at the sound of the stair."
ORX opt_01_1/rxn_01_1 -> enc_02 | O:Ask the keeper about the ledger of hours | D:C(1)
# R rxn_01_1: The keeper's cellar is dark, and the old mechanism waits in the dark for someone to understand what it was built to do.
# DIRAC opt_01_1: ["op_name_1", "c1"]
ORX opt_01_2/rxn_01_2 -> enc_02 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_01_2: The mechanism is a key, not a rope, and it fits only the door of the bell's chamber.
# DIRAC opt_01_2: ["op_name_2", "c3"]
ORX opt_01_3/rxn_01_3 -> enc_02 | O:Follow the sound of the stair to the bell's chamber | D:C(1)
# R rxn_01_3: Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair.
# DIRAC opt_01_3: ["op_name_3", "c5"]

ENC enc_02 turn=0..999
# TITLE: The Ledger of Hours
# T: The ledger of hours is open on the merchant's table, recording every coin and every bargain made in the name of the old order.
# T: 
# T: "Everything in this city is worth what someone will pay for it," said Corvin, tapping the ledger with one finger. "And the bell is no different, only louder."
# T: 
# T: "The bell is sold to a stranger and the city wakes to a foreign hour," said Maren. "But the city holds its breath at the sound of the stair."
# T: 
# T: "If the bell is cracked, then something inside it is cracked too," said Pip, tracing the crack with his thumb. "And if the clapper is missing, then the bell is missing its voice."
# T: 
# T: "The keeper's cellar is dark," said Corvin. "The old mechanism waits in the dark for someone to understand what it was built to do."
ORX opt_02_1/rxn_02_1 -> enc_03 | O:Read the ledger of hours aloud | D:C(1)
# R rxn_02_1: The ledger of hours is open on the merchant's table, recording every coin and every bargain made in the name of the old order.
# DIRAC opt_02_1: ["op_name_4", "c1"]
ORX opt_02_2/rxn_02_2 -> enc_03 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_02_2: The mechanism is a key, not a rope, and it fits only the door of the bell's chamber.
# DIRAC opt_02_2: ["op_name_5", "c3"]
ORX opt_02_3/rxn_02_3 -> enc_03 | O:Follow the sound of the stair to the bell's chamber | D:C(1)
# R rxn_02_3: Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair.
# DIRAC opt_02_3: ["op_name_6", "c5"]

ENC enc_03 turn=0..999
# TITLE: The Bell's Chamber
# T: The mechanism is a key, not a rope, and it fits only the door of the bell's chamber.
# T: 
# T: "The bell is sold to a stranger and the city wakes to a foreign hour," said Maren. "But the city holds its breath at the sound of the stair."
# T: 
# T: "If the bell is cracked, then something inside it is cracked too," said Pip, holding the mechanism up to the light. "And if the clapper is missing, then the bell is missing its voice."
# T: 
# T: "The keeper's cellar is dark," said Corvin. "The old mechanism waits in the dark for someone to understand what it was built to do."
# T: 
# T: "The bell is given away freely and the city keeps its own time without it," said Maren. "But the city holds its breath at the sound of the stair."
ORX opt_03_1/rxn_03_1 -> enc_04 | O:Open the bell's chamber with the mechanism | D:C(1)
# R rxn_03_1: The mechanism is a key, not a rope, and it fits only the door of the bell's chamber.
# DIRAC opt_03_1: ["op_name_7", "c3"]
ORX opt_03_2/rxn_03_2 -> enc_05 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_03_2: The mechanism is a key, not a rope, and it fits only the door of the bell's chamber.
# DIRAC opt_03_2: ["op_name_8", "c3"]
ORX opt_03_3/rxn_03_3 -> enc_04 | O:Follow the sound of the stair to the bell's chamber | D:C(1)
# R rxn_03_3: Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair.
# DIRAC opt_03_3: ["op_name_1", "c5"]

ENC enc_04 turn=0..999
# TITLE: The Cracked Bell at Dusk
# T: The bell in the tower has a crack running down its flank, and the keeper stands before it with a lamp in one hand and a question in the other.
# T: 
# T: "It is not broken," the keeper says, tracing the line with a fingertip. "It is cracked, and the crack is older than the tower itself."
# T: 
# T: "Then what does it mean?" asks the merchant, leaning in close.
# T: 
# T: "That something inside it is cracked too," the keeper answers, and the lamp flickers in the wind that comes down from the ridge."
ORX opt_04_1/rxn_04_1 -> enc_05 | O:Read the ledger of hours aloud | D:C(1)
# R rxn_04_1: The ledger opens with a single line: the bell was rung once, long ago, and the city did not wake.
# DIRAC opt_04_1: ["op_name_1", "c1"]
ORX opt_04_2/rxn_04_2 -> enc_05 | O:Ask the keeper what the crack means | D:C(1)
# R rxn_04_2: The keeper says the crack is a scar from the day the bell was first cast, and the city has been listening to it ever since.
# DIRAC opt_04_2: ["op_name_2", "c2"]
ORX opt_04_3/rxn_04_3 -> enc_05 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_04_3: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor.
# DIRAC opt_04_3: ["op_name_3", "c3"]

ENC enc_05 turn=0..999
# TITLE: The Ledger of Dust and Coin
# T: The merchant's table is covered in ledgers, each one open to a different page, and the room smells of ink and old paper.
# T: 
# T: "Every coin, every bargain, every name," the merchant says, tapping the top ledger with one finger. "The city has been keeping its accounts since before the tower was built."
# T: 
# T: "And the bell?" asks the keeper.
# T: 
# T: "The bell is not in the ledger," the merchant says slowly. "The bell was never a transaction. It was a promise."
ORX opt_05_1/rxn_05_1 -> enc_06 | O:Read the ledger of hours aloud | D:C(1)
# R rxn_05_1: The ledger opens with a single line: the bell was rung once, long ago, and the city did not wake.
# DIRAC opt_05_1: ["op_name_1", "c1"]
ORX opt_05_2/rxn_05_2 -> enc_06 | O:Ask the merchant what the bell means | D:C(1)
# R rxn_05_2: The merchant says the bell is the one thing in the city that cannot be bought, sold, or counted.
# DIRAC opt_05_2: ["op_name_2", "c2"]
ORX opt_05_3/rxn_05_3 -> enc_06 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_05_3: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor.
# DIRAC opt_05_3: ["op_name_3", "c3"]

ENC enc_06 turn=0..999
# TITLE: The Cracked Bell and the Silent Stair
# T: The keeper stands at the base of the stair that leads to the bell's chamber, and the wind carries the sound of a bell that is not ringing.
# T: 
# T: "The stair is empty," the keeper says. "But the bell is not silent. It is waiting."
# T: 
# T: "Waiting for what?" asks the merchant.
# T: 
# T: "For someone to understand what it was built to do," the keeper answers, and the lamp flickers in the dark.
ORX opt_06_1/rxn_06_1 -> enc_07 | O:Read the ledger of hours aloud | D:C(1)
# R rxn_06_1: The ledger opens with a single line: the bell was rung once, long ago, and the city did not wake.
# DIRAC opt_06_1: ["op_name_1", "c1"]
ORX opt_06_2/rxn_06_2 -> enc_07 | O:Ask the keeper what the crack means | D:C(1)
# R rxn_06_2: The keeper says the crack is a scar from the day the bell was first cast, and the city has been listening to it ever since.
# DIRAC opt_06_2: ["op_name_2", "c2"]
ORX opt_06_3/rxn_06_3 -> enc_07 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_06_3: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor.
# DIRAC opt_06_3: ["op_name_3", "c3"]

ENC enc_07 turn=0..999
# TITLE: The Merchant's Bargain
# T: The merchant's table is crowded with ledgers, each one open to a different page, and the room smells of ink and old paper. The bell is not in the ledger, the merchant says slowly, tapping the top ledger with one finger. The bell was never a transaction. It was a promise.
# T: 
# T: The keeper stands at the far end of the table, and the keeper's ledger holds the proof of a single, deliberate ringing, and the keeper does not want to say what it was built to do. The merchant offers to buy the bell's silence, and the room goes quiet at the offer.
# T: 
# T: The keeper says the bell is a key, not a rope, and it fits only the door of the bell's chamber. The merchant says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself.
ORX opt_07_1/rxn_07_1 -> enc_08 | O:Read the ledger of hours aloud | D:C(1)
# R rxn_07_1: The ledger opens with a single line: the bell was rung once, long ago, and the city did not wake.
# DIRAC opt_07_1: ["op_name_1", "c1"]
ORX opt_07_2/rxn_07_2 -> enc_08 | O:Ask the merchant what the bell means | D:C(1)
# R rxn_07_2: The merchant says the bell is the one thing in the city that cannot be bought, sold, or counted.
# DIRAC opt_07_2: ["op_name_2", "c2"]
ORX opt_07_3/rxn_07_3 -> enc_08 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_07_3: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor.
# DIRAC opt_07_3: ["op_name_3", "c3"]

ENC enc_08 turn=0..999
# TITLE: The Keeper's Ledger
# T: The keeper's ledger holds the proof of a single, deliberate ringing, and the keeper does not want to say what it was built to do. The keeper says the bell was never rung by accident, and the keeper's ledger holds the proof of a single, deliberate ringing.
# T: 
# T: The merchant says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself. The keeper says the bell is a key, not a rope, and it fits only the door of the bell's chamber.
# T: 
# T: The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay. The keeper says the bell is a key, not a rope, and it fits only the door of the bell's chamber.
ORX opt_08_1/rxn_08_1 -> enc_09 | O:Read the ledger of hours aloud | D:C(1)
# R rxn_08_1: The ledger opens with a single line: the bell was rung once, long ago, and the city did not wake.
# DIRAC opt_08_1: ["op_name_1", "c1"]
ORX opt_08_2/rxn_08_2 -> enc_09 | O:Ask the merchant what the bell means | D:C(1)
# R rxn_08_2: The merchant says the bell is the one thing in the city that cannot be bought, sold, or counted.
# DIRAC opt_08_2: ["op_name_2", "c2"]
ORX opt_08_3/rxn_08_3 -> enc_09 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_08_3: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor.
# DIRAC opt_08_3: ["op_name_3", "c3"]

ENC enc_09 turn=0..999
# TITLE: The Stair of Pip
# T: The keeper stands at the base of the stair that leads to the bell's chamber, and the wind carries the sound of a bell that is not ringing. The keeper says the bell is a key, not a rope, and it fits only the door of the bell's chamber.
# T: 
# T: The merchant says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself. The keeper says the bell is a key, not a rope, and it fits only the door of the bell's chamber.
# T: 
# T: The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay. The keeper says the bell is a key, not a rope, and it fits only the door of the bell's chamber.
ORX opt_09_1/rxn_09_1 -> enc_10 | O:Read the ledger of hours aloud | D:C(1)
# R rxn_09_1: The ledger opens with a single line: the bell was rung once, long ago, and the city did not wake.
# DIRAC opt_09_1: ["op_name_1", "c1"]
ORX opt_09_2/rxn_09_2 -> enc_11 | O:Ask the merchant what the bell means | D:C(1)
# R rxn_09_2: The merchant says the bell is the one thing in the city that cannot be bought, sold, or counted.
# DIRAC opt_09_2: ["op_name_2", "c2"]
ORX opt_09_3/rxn_09_3 -> enc_10 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_09_3: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor.
# DIRAC opt_09_3: ["op_name_3", "c3"]

ENC enc_10 turn=0..999
# TITLE: The Watcher's Stair
# T: The stair climbs out of the dark and into the sound of the city holding its breath. Maren stands at the foot of it, and the keeper's ledger is open in her hands, recording every coin and every bargain made in the name of the old order. The bell does not ring for the hour; it rings for the dead, and the dead do not care what the merchants call it.
# T: 
# T: The merchant says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself. The keeper says the bell is a key, not a rope, and it fits only the door of the bell's chamber.
# T: 
# T: Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair. The keeper's cellar is dark, and the old mechanism waits in the dark for someone to understand what it was built to do.
ORX opt_10_1/rxn_10_1 -> enc_11 | O:Read the ledger of hours aloud | D:C(1)
# R rxn_10_1: The ledger opens with a single line: the bell was rung once, long ago, and the city did not wake. The merchant hears the words and goes quiet, and the keeper says the bell was never rung by accident.
# DIRAC opt_10_1: ["op_name_1", "c1"]
ORX opt_10_2/rxn_10_2 -> enc_11 | O:Ask the merchant what the bell means | D:C(1)
# R rxn_10_2: The merchant says the bell is the one thing in the city that cannot be bought, sold, or counted. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
# DIRAC opt_10_2: ["op_name_2", "c2"]
ORX opt_10_3/rxn_10_3 -> enc_11 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_10_3: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair.
# DIRAC opt_10_3: ["op_name_3", "c3"]

ENC enc_11 turn=0..999
# TITLE: The Square at Midnight
# T: The square at midnight holds its breath for the answer. Three people stand where the old order once gathered, and the bell does not ring for the hour; it rings for the dead, and the dead do not care what the merchants call it. The merchant says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself.
# T: 
# T: The keeper's cellar is dark, and the old mechanism waits in the dark for someone to understand what it was built to do. The keeper says the bell is a key, not a rope, and it fits only the door of the bell's chamber.
# T: 
# T: The ledger of hours is open on the merchant's table, recording every coin and every bargain made in the name of the old order. The merchant says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself.
ORX opt_11_1/rxn_11_1 -> enc_12 | O:Read the ledger of hours aloud | D:C(1)
# R rxn_11_1: The ledger opens with a single line: the bell was rung once, long ago, and the city did not wake. The merchant hears the words and goes quiet, and the keeper says the bell was never rung by accident.
# DIRAC opt_11_1: ["op_name_1", "c1"]
ORX opt_11_2/rxn_11_2 -> enc_12 | O:Ask the merchant what the bell means | D:C(1)
# R rxn_11_2: The merchant says the bell is the one thing in the city that cannot be bought, sold, or counted. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
# DIRAC opt_11_2: ["op_name_2", "c2"]
ORX opt_11_3/rxn_11_3 -> enc_12 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_11_3: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair.
# DIRAC opt_11_3: ["op_name_3", "c3"]

ENC enc_12 turn=0..999
# TITLE: The Silence of the Merchant
# T: The silence of the merchant settles over the square like a held breath. The bell does not ring for the hour; it rings for the dead, and the dead do not care what the merchants call it. The merchant says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself.
# T: 
# T: The keeper's cellar is dark, and the old mechanism waits in the dark for someone to understand what it was built to do. The keeper says the bell is a key, not a rope, and it fits only the door of the bell's chamber.
# T: 
# T: The ledger of hours is open on the merchant's table, recording every coin and every bargain made in the name of the old order. The merchant says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself.
ORX opt_12_1/rxn_12_1 -> enc_13 | O:Read the ledger of hours aloud | D:C(1)
# R rxn_12_1: The ledger opens with a single line: the bell was rung once, long ago, and the city did not wake. The merchant hears the words and goes quiet, and the keeper says the bell was never rung by accident.
# DIRAC opt_12_1: ["op_name_1", "c1"]
ORX opt_12_2/rxn_12_2 -> enc_13 | O:Ask the merchant what the bell means | D:C(1)
# R rxn_12_2: The merchant says the bell is the one thing in the city that cannot be bought, sold, or counted. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
# DIRAC opt_12_2: ["op_name_2", "c2"]
ORX opt_12_3/rxn_12_3 -> enc_13 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_12_3: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair.
# DIRAC opt_12_3: ["op_name_3", "c3"]

ENC enc_13 turn=0..999
# TITLE: The Keeper's Silence
# T: The bell is sold to a stranger and the city wakes to a foreign hour. The keeper's ledger records the loss in a hand that has gone unsteady, and the merchant watches the keeper's face while the stranger counts coins into a velvet pouch. Maren says the bell was never meant to be owned, only kept, and she cannot bear that the city's rest is now a rumor.
# T: 
# T: The merchant says everything in this city is worth what someone will pay for it, and the bell is no different, only louder. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair. The keeper says the bell is a key, not a rope, and it fits only the door of the bell's chamber.
# T: 
# T: The keeper's cellar is dark, and the old mechanism waits in the dark for someone to understand what it was built to do. The bell does not ring for the hour; it rings for the dead, and the dead do not care what the merchants call it. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
ORX opt_13_1/rxn_13_1 -> enc_14 | O:Read the ledger of hours aloud | D:C(1)
# R rxn_13_1: The ledger opens with a single line: the bell was rung once, long ago, and the city did not wake. The merchant hears the words and goes quiet, and the keeper says the bell was never rung by accident.
# DIRAC opt_13_1: ["op_name_1", "c1"]
ORX opt_13_2/rxn_13_2 -> enc_14 | O:Ask the merchant what the bell means | D:C(1)
# R rxn_13_2: The merchant says the bell is the one thing in the city that cannot be bought, sold, or counted. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
# DIRAC opt_13_2: ["op_name_2", "c2"]
ORX opt_13_3/rxn_13_3 -> enc_14 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_13_3: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair.
# DIRAC opt_13_3: ["op_name_3", "c3"]

ENC enc_14 turn=0..999
# TITLE: The Stair of the Watcher
# T: The mechanism is a key, not a rope, and it fits only the door of the bell's chamber. The keeper stands at the base of the stair that leads to the bell's chamber, and the wind carries the sound of a bell that is not ringing. Maren says the bell was never rung by accident, and the keeper's ledger holds the proof of a single, deliberate ringing, and the keeper does not want to say what it was built to do.
# T: 
# T: The bell is cracked, not broken, in the Bell Tower at dusk. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair. The clapper is missing, not the bell, and the keeper's ledger records a single ringing long ago.
# T: 
# T: The merchant says everything in this city is worth what someone will pay for it, and the bell is no different, only louder. The keeper says the bell is a key, not a rope, and it fits only the door of the bell's chamber. The bell does not ring for the hour; it rings for the dead, and the dead do not care what the merchants call it.
ORX opt_14_1/rxn_14_1 -> enc_15 | O:Ask the keeper what the mechanism does | D:C(1)
# R rxn_14_1: The keeper says the mechanism is a key, not a rope, and it fits only the door of the bell's chamber. The merchant says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself.
# DIRAC opt_14_1: ["op_name_4", "c3"]
ORX opt_14_2/rxn_14_2 -> enc_15 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_14_2: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair.
# DIRAC opt_14_2: ["op_name_5", "c5"]
ORX opt_14_3/rxn_14_3 -> enc_15 | O:Ask the merchant what the bell means | D:C(1)
# R rxn_14_3: The merchant says the bell is the one thing in the city that cannot be bought, sold, or counted. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
# DIRAC opt_14_3: ["op_name_6", "c2"]

ENC enc_15 turn=0..999
# TITLE: The Hour of the Bell
# T: The bell rings once, and the city must decide what it heard, and the square at midnight holds its breath for the answer. The bell does not ring for the hour; it rings for the dead, and the dead do not care what the merchants call it. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
# T: 
# T: The bell is cracked, not broken, in the Bell Tower at dusk. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair. The clapper is missing, not the bell, and the keeper's ledger records a single ringing long ago.
# T: 
# T: The keeper's cellar is dark, and the old mechanism waits in the dark for someone to understand what it was built to do. The bell is sold to a stranger and the city wakes to a foreign hour, and the keeper's ledger records the loss. The merchant says everything in this city is worth what someone will pay for it, and the bell is no different, only louder.
ORX opt_15_1/rxn_15_1 -> enc_16 | O:Read the ledger of hours aloud | D:C(1)
# R rxn_15_1: The ledger opens with a single line: the bell was rung once, long ago, and the city did not wake. The merchant hears the words and goes quiet, and the keeper says the bell was never rung by accident.
# DIRAC opt_15_1: ["op_name_7", "c1"]
ORX opt_15_2/rxn_15_2 -> enc_17 | O:Ask the keeper what the mechanism does | D:C(1)
# R rxn_15_2: The keeper says the mechanism is a key, not a rope, and it fits only the door of the bell's chamber. The merchant says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself.
# DIRAC opt_15_2: ["op_name_8", "c3"]
ORX opt_15_3/rxn_15_3 -> enc_16 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_15_3: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair.
# DIRAC opt_15_3: ["op_name_1", "c5"]

ENC enc_16 turn=0..999
# TITLE: The Merchant's Silence
# T: The bell is sold to a stranger and the city wakes to a foreign hour, and the merchant's ledger records the loss. Corvin stands at the table where the ledger of hours lies open, every coin and every bargain made in the name of the old order recorded in his own hand. He does not speak. The bell does not ring for the hour; it rings for the dead, and the dead do not care what the merchants call it.
# T: 
# T: Maren watches the merchant's silence and understands that the bell was never meant to be owned, only kept. The keeper's cellar is dark, and the old mechanism waits in the dark for someone to understand what it was built to do. The bell is cracked, not broken, in the Bell Tower at dusk, and the clapper is missing, not the bell.
# T: 
# T: Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair. The merchant says everything in this city is worth what someone will pay for it, and the bell is no different, only louder. But the keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
ORX opt_16_1/rxn_16_1 -> enc_17 | O:Ask the merchant what the bell means | D:C(1)
# R rxn_16_1: The merchant says the bell is the one thing in the city that cannot be bought, sold, or counted. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
# DIRAC opt_16_1: ["op_name_1", "c1"]
ORX opt_16_2/rxn_16_2 -> enc_17 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_16_2: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair.
# DIRAC opt_16_2: ["op_name_2", "c5"]
ORX opt_16_3/rxn_16_3 -> enc_17 | O:Read the ledger of hours aloud | D:C(1)
# R rxn_16_3: The ledger opens with a single line: the bell was rung once, long ago, and the city did not wake. The merchant hears the words and goes quiet, and the keeper says the bell was never rung by accident.
# DIRAC opt_16_3: ["op_name_3", "c1"]

ENC enc_17 turn=0..999
# TITLE: The Keeper's Stair
# T: The mechanism is a key, not a rope, and it fits only the door of the bell's chamber. The keeper stands at the base of the stair that leads to the bell's chamber, and the wind carries the sound of a bell that is not ringing. Maren says the bell was never rung by accident, and the keeper's ledger holds the proof of a single, deliberate ringing, and the keeper does not want to say what it was built to do.
# T: 
# T: The bell is cracked, not broken, in the Bell Tower at dusk. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair. The clapper is missing, not the bell, and the keeper's ledger records a single ringing long ago.
# T: 
# T: The keeper's cellar is dark, and the old mechanism waits in the dark for someone to understand what it was built to do. The bell is sold to a stranger and the city wakes to a foreign hour, and the keeper's ledger records the loss. The merchant says everything in this city is worth what someone will pay for it, and the bell is no different, only louder.
ORX opt_17_1/rxn_17_1 -> enc_18 | O:Ask the keeper what the mechanism does | D:C(1)
# R rxn_17_1: The keeper says the mechanism is a key, not a rope, and it fits only the door of the bell's chamber. The merchant says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself.
# DIRAC opt_17_1: ["op_name_4", "c3"]
ORX opt_17_2/rxn_17_2 -> enc_18 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_17_2: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair.
# DIRAC opt_17_2: ["op_name_5", "c5"]
ORX opt_17_3/rxn_17_3 -> enc_18 | O:Read the ledger of hours aloud | D:C(1)
# R rxn_17_3: The ledger opens with a single line: the bell was rung once, long ago, and the city did not wake. The merchant hears the words and goes quiet, and the keeper says the bell was never rung by accident.
# DIRAC opt_17_3: ["op_name_6", "c1"]

ENC enc_18 turn=0..999
# TITLE: The Watcher's Hour
# T: All three must choose what the bell means, and the square at midnight holds its breath for the answer. The bell does not ring for the hour; it rings for the dead, and the dead do not care what the merchants call it. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
# T: 
# T: The bell is cracked, not broken, in the Bell Tower at dusk. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair. The clapper is missing, not the bell, and the keeper's ledger records a single ringing long ago.
# T: 
# T: The keeper's cellar is dark, and the old mechanism waits in the dark for someone to understand what it was built to do. The bell is sold to a stranger and the city wakes to a foreign hour, and the keeper's ledger records the loss. The merchant says everything in this city is worth what someone will pay for it, and the bell is no different, only louder.
ORX opt_18_1/rxn_18_1 -> enc_19 | O:Ask the merchant what the bell means | D:C(1)
# R rxn_18_1: The merchant says the bell is the one thing in the city that cannot be bought, sold, or counted. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
# DIRAC opt_18_1: ["op_name_7", "c1"]
ORX opt_18_2/rxn_18_2 -> enc_19 | O:Ask the keeper what the mechanism does | D:C(1)
# R rxn_18_2: The keeper says the mechanism is a key, not a rope, and it fits only the door of the bell's chamber. The merchant says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself.
# DIRAC opt_18_2: ["op_name_8", "c3"]
ORX opt_18_3/rxn_18_3 -> enc_19 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_18_3: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair.
# DIRAC opt_18_3: ["op_name_1", "c5"]

ENC enc_19 turn=0..999
# TITLE: The Merchant's Hour
# T: The bell is sold to a stranger and the city wakes to a foreign hour, and the merchant's ledger records the loss. Corvin stands at the table where the ledger of hours lies open, his fingers resting on the line where the bell was entered as a commodity. He says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself.
# T: 
# T: The merchant says everything in this city is worth what someone will pay for it, and the bell is no different, only louder. Pip watches the merchant's hand hover over the ledger, and the child says if the bell is cracked, then something inside it is cracked too, and if the clapper is missing, then the bell is missing its voice.
# T: 
# T: The merchant says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself. The merchant says everything in this city is worth what someone will pay for it, and the bell is no different, only louder.
ORX opt_19_1/rxn_19_1 -> enc_20 | O:Ask the merchant what the bell means | D:C(1)
# R rxn_19_1: The merchant says the bell is the one thing in the city that cannot be bought, sold, or counted. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
# DIRAC opt_19_1: ["op_name_7", "c1"]
ORX opt_19_2/rxn_19_2 -> enc_20 | O:Ask the keeper what the mechanism does | D:C(1)
# R rxn_19_2: The keeper says the mechanism is a key, not a rope, and it fits only the door of the bell's chamber. The merchant says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself.
# DIRAC opt_19_2: ["op_name_8", "c3"]
ORX opt_19_3/rxn_19_3 -> enc_20 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_19_3: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair.
# DIRAC opt_19_3: ["op_name_1", "c5"]

ENC enc_20 turn=0..999
# TITLE: The Keeper's Hour
# T: The bell is sold to a stranger and the city wakes to a foreign hour, and the keeper's ledger records the loss. Maren stands at the base of the stair that leads to the bell's chamber, and the wind carries the sound of a bell that is not ringing. She says the bell was never rung by accident, and the keeper's ledger holds the proof of a single, deliberate ringing, and the keeper does not want to say what it was built to do.
# T: 
# T: The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
# T: 
# T: The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
ORX opt_20_1/rxn_20_1 -> enc_21 | O:Ask the keeper what the mechanism does | D:C(1)
# R rxn_20_1: The keeper says the mechanism is a key, not a rope, and it fits only the door of the bell's chamber. The merchant says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself.
# DIRAC opt_20_1: ["op_name_4", "c3"]
ORX opt_20_2/rxn_20_2 -> enc_21 | O:Ask the merchant what the bell means | D:C(1)
# R rxn_20_2: The merchant says the bell is the one thing in the city that cannot be bought, sold, or counted. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
# DIRAC opt_20_2: ["op_name_7", "c1"]
ORX opt_20_3/rxn_20_3 -> enc_21 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_20_3: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair.
# DIRAC opt_20_3: ["op_name_5", "c5"]

ENC enc_21 turn=0..999
# TITLE: The Stair of the Keeper
# T: The mechanism is a key, not a rope, and it fits only the door of the bell's chamber. The keeper stands at the base of the stair that leads to the bell's chamber, and the wind carries the sound of a bell that is not ringing. Maren says the bell was never rung by accident, and the keeper's ledger holds the proof of a single, deliberate ringing, and the keeper does not want to say what it was built to do.
# T: 
# T: The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
# T: 
# T: The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
ORX opt_21_1/rxn_21_1 -> enc_22 | O:Ask the keeper what the mechanism does | D:C(1)
# R rxn_21_1: The keeper says the mechanism is a key, not a rope, and it fits only the door of the bell's chamber. The merchant says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself.
# DIRAC opt_21_1: ["op_name_4", "c3"]
ORX opt_21_2/rxn_21_2 -> enc_23 | O:Ask the merchant what the bell means | D:C(1)
# R rxn_21_2: The merchant says the bell is the one thing in the city that cannot be bought, sold, or counted. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
# DIRAC opt_21_2: ["op_name_7", "c1"]
ORX opt_21_3/rxn_21_3 -> enc_22 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_21_3: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair.
# DIRAC opt_21_3: ["op_name_5", "c5"]

ENC enc_22 turn=0..999
# TITLE: The Hour of the Watcher
# T: The square at midnight holds its breath for the answer. Maren stands at the base of the stair that leads to the bell's chamber, and the wind carries the sound of a bell that is not ringing. She says the bell was never rung by accident, and the keeper's ledger holds the proof of a single, deliberate ringing, and the keeper does not want to say what it was built to do.
# T: 
# T: Corvin stands beside her, turning the ledger of hours over in his hands as if it were a ledger of coin. He says everything in this city is worth what someone will pay for it, and the bell is no different, only louder. Pip, small and barefoot on the stair, says if the bell is cracked, then something inside it is cracked too, and if the clapper is missing, then the bell is missing its voice.
# T: 
# T: The three of them look at one another across the dark square. The bell is sold to a stranger and the city wakes to a foreign hour, and none of them can say what the bell was built to do. The keeper's cellar is dark, and the old mechanism waits in the dark for someone to understand what it was built to do.
ORX opt_22_1/rxn_22_1 -> enc_23 | O:Ask the keeper what the mechanism does | D:C(1)
# R rxn_22_1: The keeper says the mechanism is a key, not a rope, and it fits only the door of the bell's chamber. The merchant says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself.
# DIRAC opt_22_1: ["op_name_4", "c3"]
ORX opt_22_2/rxn_22_2 -> enc_23 | O:Ask the merchant what the bell means | D:C(1)
# R rxn_22_2: The merchant says the bell is the one thing in the city that cannot be bought, sold, or counted. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
# DIRAC opt_22_2: ["op_name_7", "c1"]
ORX opt_22_3/rxn_22_3 -> enc_23 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_22_3: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair.
# DIRAC opt_22_3: ["op_name_5", "c5"]

ENC enc_23 turn=0..999
# TITLE: The Merchant's Stair
# T: The mechanism is a key, not a rope, and it fits only the door of the bell's chamber. The keeper stands at the base of the stair that leads to the bell's chamber, and the wind carries the sound of a bell that is not ringing. Maren says the bell was never rung by accident, and the keeper's ledger holds the proof of a single, deliberate ringing, and the keeper does not want to say what it was built to do.
# T: 
# T: Corvin climbs the stair with the ledger of hours open on the merchant's table, recording every coin and every bargain made in the name of the old order. He says everything in this city is worth what someone will pay for it, and the bell is no different, only louder. Pip, small and barefoot on the stair, says if the bell is cracked, then something inside it is cracked too, and if the clapper is missing, then the bell is missing its voice.
# T: 
# T: The three of them look at one another across the dark square. The bell is sold to a stranger and the city wakes to a foreign hour, and none of them can say what the bell was built to do. The keeper's cellar is dark, and the old mechanism waits in the dark for someone to understand what it was built to do.
ORX opt_23_1/rxn_23_1 -> enc_24 | O:Ask the keeper what the mechanism does | D:C(1)
# R rxn_23_1: The keeper says the mechanism is a key, not a rope, and it fits only the door of the bell's chamber. The merchant says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself.
# DIRAC opt_23_1: ["op_name_4", "c3"]
ORX opt_23_2/rxn_23_2 -> enc_24 | O:Ask the merchant what the bell means | D:C(1)
# R rxn_23_2: The merchant says the bell is the one thing in the city that cannot be bought, sold, or counted. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
# DIRAC opt_23_2: ["op_name_7", "c1"]
ORX opt_23_3/rxn_23_3 -> enc_24 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_23_3: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair.
# DIRAC opt_23_3: ["op_name_5", "c5"]

ENC enc_24 turn=0..999
# TITLE: The Keeper's Hour of the Bell
# T: The bell rings once, and the city must decide what it heard, and the square at midnight holds its breath for the answer. Maren stands at the base of the stair that leads to the bell's chamber, and the wind carries the sound of a bell that is not ringing. She says the bell was never rung by accident, and the keeper's ledger holds the proof of a single, deliberate ringing, and the keeper does not want to say what it was built to do.
# T: 
# T: Corvin stands beside her, turning the ledger of hours over in his hands as if it were a ledger of coin. He says everything in this city is worth what someone will pay for it, and the bell is no different, only louder. Pip, small and barefoot on the stair, says if the bell is cracked, then something inside it is cracked too, and if the clapper is missing, then the bell is missing its voice.
# T: 
# T: The three of them look at one another across the dark square. The bell is sold to a stranger and the city wakes to a foreign hour, and none of them can say what the bell was built to do. The keeper's cellar is dark, and the old mechanism waits in the dark for someone to understand what it was built to do.
ORX opt_24_1/rxn_24_1 -> enc_25 | O:Ask the keeper what the mechanism does | D:C(1)
# R rxn_24_1: The keeper says the mechanism is a key, not a rope, and it fits only the door of the bell's chamber. The merchant says the bell is a commodity whose value is only what the highest bidder will pay, and the keeper says the bell is a promise the city made to itself.
# DIRAC opt_24_1: ["op_name_4", "c3"]
ORX opt_24_2/rxn_24_2 -> enc_25 | O:Ask the merchant what the bell means | D:C(1)
# R rxn_24_2: The merchant says the bell is the one thing in the city that cannot be bought, sold, or counted. The keeper says the bell is a promise the city made to itself, and the merchant says the bell is a commodity whose value is only what the highest bidder will pay.
# DIRAC opt_24_2: ["op_name_7", "c1"]
ORX opt_24_3/rxn_24_3 -> enc_25 | O:Search the bell's chamber for the mechanism | D:C(1)
# R rxn_24_3: The chamber holds nothing but dust and a single key, small as a fingernail, resting on the floor. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair.
# DIRAC opt_24_3: ["op_name_5", "c5"]

ENC enc_25 turn=0..999
# TITLE: The Watcher's Hour of the Bell
# T: The square at midnight holds its breath for the answer. Maren stands at the base of the stair that leads to the bell's chamber, and the wind carries the smell of rain and old brass. Around her, the city has gathered in the dark: the merchants with their ledgers, the children with their questions, and the stranger who bought the bell and does not know what it is for.
# T: 
# T: "The bell does not ring for the hour," Maren says, not looking up from the cracked bell she has been holding. "It rings for the dead, and the dead do not care what the merchants call it."
# T: 
# T: "Then why does the keeper's ledger record a single ringing long ago?" asks Pip, clutching his fists. "If the bell is cracked, then something inside it is cracked too, and if the clapper is missing, then the bell is missing its voice."
# T: 
# T: "Because the bell is missing its voice," Maren says, and the square goes quiet. "And the dead are waiting for it to find it again."
ORX opt_25_1/rxn_25_1 -> enc_26 | O:Ask the stranger what the bell was sold for | D:C(1)
# R rxn_25_1: "I bought it for the sound," the stranger says, "not for what it means. The bell is sold to a stranger and the city wakes to a foreign hour."
# DIRAC opt_25_1: ["op_name_1", "c0"]
ORX opt_25_2/rxn_25_2 -> enc_26 | O:Tell Pip the bell is cracked, not broken | D:C(1)
# R rxn_25_2: "The bell is found cracked, not broken, in the Bell Tower at dusk," Maren says, "and the clapper is missing, not the bell. Something inside it is cracked too."
# DIRAC opt_25_2: ["op_name_2", "c4"]
ORX opt_25_3/rxn_25_3 -> enc_26 | O:Ask Corvin what the bell is worth | D:C(1)
# R rxn_25_3: "Everything in this city is worth what someone will pay for it, and the bell is no different, only louder," Corvin says, and the square laughs.
# DIRAC opt_25_3: ["op_name_3", null]

ENC enc_26 turn=0..999
# TITLE: The Merchant's Hour of the Bell
# T: The square at midnight holds its breath for the answer. Corvin stands at the center of the square, his ledger open on the merchant's table, recording every coin and every bargain made in the name of the old order. The bell is not here, but its absence hangs over the crowd like a held breath.
# T: 
# T: "The bell is a commodity whose value is only what the highest bidder will pay," Corvin says, tapping the ledger with one finger. "He cannot bear that it might mean something beyond price."
# T: 
# T: "And if the bell is sold to a stranger and the city wakes to a foreign hour?" asks Pip. "Then what does the city wake to?"
# T: 
# T: "The city wakes to a foreign hour," Corvin says, "and the dead are called home by a bell that does not know their names." He looks at Maren. "The bell is sold to a stranger and the city wakes to a foreign hour."
ORX opt_26_1/rxn_26_1 -> enc_27 | O:Offer to buy the bell back at any price | D:C(1)
# R rxn_26_1: "I will pay any price," Corvin says, "but the bell is sold to a stranger and the city wakes to a foreign hour. The bell is sold to a stranger and the city wakes to a foreign hour."
# DIRAC opt_26_1: ["op_name_4", null]
ORX opt_26_2/rxn_26_2 -> enc_27 | O:Ask Corvin what the bell is worth to the dead | D:C(1)
# R rxn_26_2: "The dead do not care what the merchants call it," Maren says, "and the bell is sold to a stranger and the city wakes to a foreign hour."
# DIRAC opt_26_2: ["op_name_5", "c1"]
ORX opt_26_3/rxn_26_3 -> enc_27 | O:Tell Corvin the bell is not for sale | D:C(1)
# R rxn_26_3: "The bell is sold to a stranger and the city wakes to a foreign hour," Corvin says, "but the bell is not for sale. The bell is sold to a stranger and the city wakes to a foreign hour."
# DIRAC opt_26_3: ["op_name_6", null]

ENC enc_27 turn=0..999
# TITLE: The Keeper's Stair of the Bell
# T: The square at midnight holds its breath for the answer. Maren stands at the base of the stair that leads to the bell's chamber, and the wind carries the smell of rain and old brass. The mechanism is a key, not a rope, and it fits only the door of the bell's chamber.
# T: 
# T: "The mechanism is a key, not a rope, and it fits only the door of the bell's chamber," Maren says, looking up the stair. "The keeper's cellar is dark, and the old mechanism waits in the dark for someone to understand what it was built to do."
# T: 
# T: "And if the bell is cracked, then something inside it is cracked too, and if the clapper is missing, then the bell is missing its voice," Pip says, climbing the stair. "Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair."
# T: 
# T: "Then we go up," Maren says, "and we find what the bell was built to do."
ORX opt_27_1/rxn_27_1 -> enc_29 | O:Climb the stair to the bell's chamber | D:C(1)
# R rxn_27_1: "Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair," Maren says.
# DIRAC opt_27_1: ["op_name_7", "c5"]
ORX opt_27_2/rxn_27_2 -> enc_28 | O:Ask the keeper what the mechanism was built to do | D:C(1)
# R rxn_27_2: "The keeper's cellar is dark, and the old mechanism waits in the dark for someone to understand what it was built to do," Maren says.
# DIRAC opt_27_2: ["op_name_8", "c2"]
ORX opt_27_3/rxn_27_3 -> enc_28 | O:Tell Pip the mechanism is a key, not a rope | D:C(1)
# R rxn_27_3: "The mechanism is a key, not a rope, and it fits only the door of the bell's chamber," Maren says.
# DIRAC opt_27_3: ["op_name_1", "c3"]

ENC enc_28 turn=0..999
# TITLE: The Watcher's Stair of the Bell
# T: The stair to the bell's chamber is narrow, worn smooth by generations of feet, and at its top the mechanism waits in the dark for someone to understand what it was built to do. Maren stands at the base of the stair, her hand resting on the cold stone, and the wind carries the smell of rain and old brass. Corvin follows her up, his ledger open in his hand, and Pip clings to the railing, his eyes wide with the certainty that something inside the bell is cracked.
# T: 
# T: The mechanism is a key, not a rope, and it fits only the door of the bell's chamber. It does not pull; it turns. It does not open; it remembers. Maren understands that the bell was never meant to be owned, only kept, and that the door of the chamber is the only place the bell can be whole again. Corvin sees a lock, a mechanism, a thing that can be bought and sold, and he reaches for it with the same hand that reaches for every other valuable object in his life. Pip sees a crack, a missing clapper, a voice that has been taken from the bell.
# T: 
# T: The square at midnight holds its breath for the answer. The three of them stand at the top of the stair, looking down at the dark square below, and the question hangs in the air like the smell of rain: what does the bell mean, and what does it want to say?
ORX opt_28_1/rxn_28_1 -> enc_29 | O:Maren turns the key and opens the chamber door | D:C(1)
# R rxn_28_1: Maren: "The bell was never meant to be owned, only kept. I will open the door and let the bell speak for itself."
# DIRAC opt_28_1: ["op_name_1", "c3"]
ORX opt_28_2/rxn_28_2 -> enc_29 | O:Corvin tries to buy the mechanism from the keeper | D:C(1)
# R rxn_28_2: Corvin: "Everything in this city is worth what someone will pay for it, and the bell is no different, only louder. I will pay whatever it takes to own this key."
# DIRAC opt_28_2: ["op_name_2", "c2"]
ORX opt_28_3/rxn_28_3 -> enc_29 | O:Pip climbs to the chamber and finds the mechanism | D:C(1)
# R rxn_28_3: Pip: "If the bell is cracked, then something inside it is cracked too, and if the clapper is missing, then the bell is missing its voice. I will find what it is hiding."
# DIRAC opt_28_3: ["op_name_3", "c5"]

ENC enc_29 turn=0..999
# TITLE: The Merchant's Hour of the Watcher
# T: The square at midnight holds its breath for the answer. Corvin stands at the center of the square, his ledger open on the merchant's table, recording every coin and every bargain made in the name of the old order. The bell is gone, and the city cannot tell when the day ends, when the dead are called home, or when the harbor opens. The three of them stand at the edge of the square, each with a different idea of what the bell means.
# T: 
# T: Corvin believes the bell is a commodity whose value is only what the highest bidder will pay. He cannot bear that it might mean something beyond price. He stands before the empty space where the bell hung, and he sees only a gap, a hole, a thing that can be filled with money. Maren stands beside him, and she sees the bell as a thing that was kept, not owned, and she cannot bear that the city's rest is now a rumor. Pip stands beside her, and he sees the bell as a thing that was hiding something, and he cannot bear that the city would not ask what it was hiding.
# T: 
# T: The square at midnight holds its breath for the answer. The three of them must choose what the bell means, and the choice will determine what the bell heard at midnight.
ORX opt_29_1/rxn_29_1 -> enc_30 | O:Maren says the bell rings for the dead | D:C(1)
# R rxn_29_1: Maren: "The bell does not ring for the hour; it rings for the dead, and the dead do not care what the merchants call it. We must let the bell ring for them."
# DIRAC opt_29_1: ["op_name_4", "c4"]
ORX opt_29_2/rxn_29_2 -> enc_30 | O:Corvin says the bell is worth what someone will pay | D:C(1)
# R rxn_29_2: Corvin: "Everything in this city is worth what someone will pay for it, and the bell is no different, only louder. We must sell the bell and let the highest bidder decide what it means."
# DIRAC opt_29_2: ["op_name_5", "c1"]
ORX opt_29_3/rxn_29_3 -> enc_30 | O:Pip says the bell is hiding something | D:C(1)
# R rxn_29_3: Pip: "If the bell is cracked, then something inside it is cracked too, and if the clapper is missing, then the bell is missing its voice. We must ask what the bell is hiding."
# DIRAC opt_29_3: ["op_name_6", "c0"]

ENC enc_30 turn=0..999
# TITLE: The Keeper's Hour of the Watcher
# T: The square at midnight holds its breath for the answer. The three of them stand at the center of the square, and the question hangs in the air like the smell of rain: what does the bell mean, and what does it want to say? The bell is gone, and the city cannot tell when the day ends, when the dead are called home, or when the harbor opens. The three of them must choose what the bell means, and the choice will determine what the bell heard at midnight.
# T: 
# T: Maren believes the bell was never meant to be owned, only kept. She cannot bear that the city's rest is now a rumor. She stands before the empty space where the bell hung, and she sees the bell as a thing that was kept, not owned, and she knows that the bell must ring for the dead. Corvin believes the bell is a commodity whose value is only what the highest bidder will pay. He cannot bear that it might mean something beyond price. He stands before the empty space where the bell hung, and he sees only a gap, a hole, a thing that can be filled with money. Pip believes the bell went missing because it was hiding something. He cannot bear that the city would not ask what it was hiding. He stands before the empty space where the bell hung, and he sees a crack, a missing clapper, a voice that has been taken from the bell.
# T: 
# T: The bell rings once, and the city must decide what it heard. The three of them stand at the center of the square, and the question hangs in the air like the smell of rain: what does the bell mean, and what does it want to say?
ORX opt_30_1/rxn_30_1 -> enc_finale | O:Maren says the bell rings for the dead | D:C(1)
# R rxn_30_1: Maren: "The bell does not ring for the hour; it rings for the dead, and the dead do not care what the merchants call it. We must let the bell ring for them."
# DIRAC opt_30_1: ["op_name_7", "c4"]
ORX opt_30_2/rxn_30_2 -> enc_finale | O:Corvin says the bell is worth what someone will pay | D:C(1)
# R rxn_30_2: Corvin: "Everything in this city is worth what someone will pay for it, and the bell is no different, only louder. We must sell the bell and let the highest bidder decide what it means."
# DIRAC opt_30_2: ["op_name_8", "c1"]
ORX opt_30_3/rxn_30_3 -> enc_finale | O:Pip says the bell is hiding something | D:C(1)
# R rxn_30_3: Pip: "If the bell is cracked, then something inside it is cracked too, and if the clapper is missing, then the bell is missing its voice. We must ask what the bell is hiding."
# DIRAC opt_30_3: ["op_name_1", "c0"]

ENC enc_finale turn=0..999
# TITLE: Act V: The Hour of the Bell
ORX opt_finale_1/rxn_finale_1 -> end_bell_returns | O:The Bell Returns | D:C(1)
ORX opt_finale_2/rxn_finale_2 -> end_bell_sold | O:The Bell Sold | D:C(1)
ORX opt_finale_3/rxn_finale_3 -> end_bell_lost_forever | O:The Bell Lost Forever | D:C(1)
ORX opt_finale_4/rxn_finale_4 -> end_bell_as_question | O:The Bell as a Question | D:C(1)
# SECRET opt_finale_4: {"requires": ["c0", "c2", "c4", "c5"], "projectors": [["X0+", 0.25, 0.95], ["01", 0.1, 0.8], ["Y1-", 0.3, 0.7]]}
ORX opt_finale_5/rxn_finale_5 -> end_bell_as_answer | O:The Bell as an Answer | D:C(1)
# SECRET opt_finale_5: {"requires": ["c1", "c3", "c4"], "projectors": [["X1+", 0.2, 0.9], ["10", 0.15, 0.75], ["Y0+", 0.35, 0.65]]}

ENC end_bell_returns turn=0..999
# TITLE: The Bell Returns
# T: The bell comes back to the city, whole and cracked, and the dead are called home. Maren stands at the base of the tower as the mechanism turns in the dark, and the clapper, missing for so long, finds its place at last. The city wakes to a sound it has not heard in a generation, and every soul in Vellum knows that the bell was never a commodity but a door.
# T: 
# T: The dead are called home, and the ledger of hours is closed for good. Corvin watches from the merchant's row, and for the first time in his life he does not reach for a price. The bell rings once more, and the city learns that some things are kept, not owned, and that the sound of a bell is not a sound to be sold but a sound to be answered.

ENC end_bell_sold turn=0..999
# TITLE: The Bell Sold
# T: Corvin sells the bell's sound to a foreign power, and the city wakes to a stranger's hour. The ledger of hours records every coin and every bargain, and the mechanism, a key that fits only the door of the bell's chamber, is left behind in the dark. The city wakes to a foreign hour, and the bell's voice is gone, carried away by men who do not know what it was built to say.
# T: 
# T: The dead are not called home, and the city learns to live without its bell. Maren watches from the cellar, and the keeper's ledger records the loss. The bell is sold, and the city wakes to a stranger's hour, and the sound of the bell is a sound that belongs to someone else now.

ENC end_bell_lost_forever turn=0..999
# TITLE: The Bell Lost Forever
# T: The bell is lost to the sea, and the city learns to keep its own time. The cracked bell is found at dusk, and the mechanism waits in the dark for someone to understand what it was built to do. The keeper's ledger records a single ringing long ago, and the clapper is missing, not the bell. The city learns to keep its own time, and the bell is gone, lost to the sea.
# T: 
# T: The city learns to keep its own time, and the dead are not called home. Maren watches from the cellar, and the keeper's ledger records the loss. The bell is lost to the sea, and the city learns to keep its own time, and the sound of the bell is a sound that belongs to the deep now.

ENC end_bell_as_question turn=0..999
# TITLE: The Bell as a Question
# T: The bell is rung as a question, and the city must answer it together. Pip climbs the stair to the bell's chamber and finds the mechanism, and the city holds its breath at the sound of the stair. The cracked bell is found at dusk, and the clapper is missing, not the bell. The bell is rung as a question, and the city must answer it together.
# T: 
# T: The city answers the bell's question, and the dead are called home. Maren stands at the base of the tower, and the mechanism turns in the dark. The bell is rung as a question, and the city must answer it together, and the sound of the bell is a sound that belongs to the city now.

ENC end_bell_as_answer turn=0..999
# TITLE: The Bell as an Answer
# T: The bell is rung as an answer, and the city must decide what it was hiding. The ledger of hours records every coin and every bargain, and the mechanism is a key, not a rope, fitting only the door of the bell's chamber. The bell is rung as an answer, and the city must decide what it was hiding.
# T: 
# T: The city decides what the bell was hiding, and the dead are called home. Maren stands at the base of the tower, and the mechanism turns in the dark. The bell is rung as an answer, and the city must decide what it was hiding, and the sound of the bell is a sound that belongs to the city now.
