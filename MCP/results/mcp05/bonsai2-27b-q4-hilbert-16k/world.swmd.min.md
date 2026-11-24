---
title: "The City That Lost Its Bell"
storyworld_id: SW-INKY-HILBERT-16384-20261124
source_format: SWMD-0-MIN
hilbert_player_data: storyworld.hilbert.json
state_variables: ["real_00", "real_01", "real_10", "real_11", "imag_00", "imag_01", "imag_10", "imag_11", "c0", "c1", "c2", "c3", "c4", "c5"]
endings:
  - id: end_restore
    type: public
    condition: "The bell rings again; order returns but the city forgets why it was ever silent."
    clue_chain: "seal -> ledger page -> counter-bell"
  - id: end_silence
    type: public
    condition: "The city embraces quiet; a new ritual forms around absence."
    clue_chain: "market trade -> borrowed hours -> plaza echo"
  - id: end_expose
    type: public
    condition: "Sable's exposure shatters the old order; the city rebuilds from raw memory."
    clue_chain: "torn page -> undercroft -> name on clapper"
  - id: end_mix
    type: secret
    condition: "The player mixes the two bells' tones, creating a sound that is neither old nor new."
    clue_chain: "seal + torn page -> counter-bell + market trade"
  - id: end_phase
    type: secret
    condition: "The player acts in a different order than the city expects, revealing that silence was a phase, not a loss."
    clue_chain: "ledger page first -> seal second -> name last"
---

# SWMD-0-MIN
id: SW-INKY-HILBERT-16384-20261124
title: The City That Lost Its Bell

ENC enc_01 turn=0..999
# TITLE: Dusk at the Silent Tower
# T: The last light thins over Vespera as Mara climbs the tower steps, her notebook already open. The public notice is nailed to the door in a clerk's careful hand: 'The bell is broken. Do not attempt repair.' She reads it twice, then looks up at the dark mouth of the tower where no sound has come for three nights.
# T: 
# T: A man in a merchant's coat steps out from the alley and watches her with the patience of someone who has been waiting for an audience. 'You're the one who keeps asking questions,' he says. 'The city is resting. Let it rest.' His voice is soft, almost kind, but his eyes are counting her.
# T: 
# T: Mara touches the notice. The paper is dry, the ink still fresh. 'Resting,' she repeats. 'Or hiding?' She glances at the door, where a faint shadow of wax catches the dying light.
ORX opt_01_1/rxn_01_1 -> enc_02 | O:Ask the merchant about the silence | D:C(1)
# R rxn_01_1: "If the bell is broken, why is the notice so new?"
# DIRAC opt_01_1: ["plaza_echo", "c5"]
ORX opt_01_2/rxn_01_2 -> enc_02 | O:Examine the tower door closely | D:C(1)
# R rxn_01_2: "I don't need your permission to look at a door."
# DIRAC opt_01_2: ["seal_check", "c0"]
ORX opt_01_3/rxn_01_3 -> enc_02 | O:Step back and listen for the bell | D:C(1)
# R rxn_01_3: "If it's broken, I should hear something."
# DIRAC opt_01_3: ["undercroft_bell", null]

ENC enc_02 turn=0..999
# TITLE: The Wax Seal on the Door
# T: Mara's fingers find the wax before the merchant can finish his sentence. It is small, round, and pressed with a mark she has never seen in any guild registry. She pulls out her notebook and begins to copy it, the graphite scratching against the page.
# T: 
# T: The merchant leans closer. 'You're copying it,' he says. Not a question. 'That's not how this works. You don't take things from the tower.' His voice drops, and for a moment she hears something underneath it: fear, or maybe guilt.
# T: 
# T: Mara does not stop drawing. The seal is not a guild mark. It is too personal, too particular. She thinks of the second bell, the one with her name on its clapper, and wonders if this is the same hand that locked the door.
ORX opt_02_1/rxn_02_1 -> enc_03 | O:Show the merchant the copy | D:C(1)
# R rxn_02_1: "I'm not hiding it. Look at it."
# DIRAC opt_02_1: ["name_on_clapper", "c0"]
ORX opt_02_2/rxn_02_2 -> enc_03 | O:Pocket the notebook and step away | D:C(1)
# R rxn_02_2: "I'll be back when you're ready to talk."
# DIRAC opt_02_2: ["market_trade", "c1"]
ORX opt_02_3/rxn_02_3 -> enc_03 | O:Press the seal against the door frame | D:C(1)
# R rxn_02_3: "If it matches, I know who was here."
# DIRAC opt_02_3: ["ledger_tear", "c2"]

ENC enc_03 turn=0..999
# TITLE: The Bell That Never Rang
# T: Inside the tower the great bell hangs in its frame, intact and cold. Its bronze skin is unmarked, but a scratch mark on the iron frame tells a different story: someone locked this door from the inside. The silence here is not empty; it is full of held breath.
# T: 
# T: Mara's hand finds the smaller bell in the undercroft, the one with her name inscribed on its clapper. She turns it over in her palm. It is warm, as if it has been waiting for her. Above, the great bell does not ring. Below, the small one seems to hum.
# T: 
# T: The merchant's voice comes from the stairwell, distant and careful. 'You can't take this. The city needs its story.' But his words sound thin now, like a page torn from a ledger that no longer exists.
ORX opt_03_1/rxn_03_1 -> enc_04 | O:Ring the small bell once | D:C(1)
# R rxn_03_1: "If I can hear it, the city can too."
# DIRAC opt_03_1: ["undercroft_bell", "c3"]
ORX opt_03_2/rxn_03_2 -> enc_05 | O:Take the ledger page from the shelf | D:C(1)
# R rxn_03_2: "This is what they tore out. I'm putting it back."
# DIRAC opt_03_2: ["ledger_tear", "c2"]
ORX opt_03_3/rxn_03_3 -> enc_04 | O:Leave the tower and let the city decide | D:C(1)
# R rxn_03_3: "Maybe the silence is the message."
# DIRAC opt_03_3: ["plaza_echo", "c4"]

ENC enc_04 turn=0..999
# TITLE: The First Quiet
# T: The city guard leans against the tower base, his hand still resting on the hilt of a sword he has not drawn in three days. "Three nights," he says, voice low, as if the words might wake something. "The bell stopped at the third watch. No one saw who touched it. No one heard it stop. The streets just... held their breath."
# T: 
# T: Mara studies the tower door, the wax seal still fresh on the wood. "If no one saw it stop, then the silence is not a theft. It is a choice. And choices leave marks."
# T: 
# T: The guard shifts his weight. "I have walked these streets for twenty years. I know every sound they make. This quiet is not empty. It is full of something I cannot name."
ORX opt_04_1/rxn_04_1 -> enc_05 | O:Ask the guard what he heard in the silence | D:C(1)
# R rxn_04_1: The guard closes his eyes. "Nothing. That is the worst part. Nothing at all. The city does not know how to be still."
# DIRAC opt_04_1: ["seal_check", "c0"]
ORX opt_04_2/rxn_04_2 -> enc_05 | O:Press him about who locked the tower door | D:C(1)
# R rxn_04_2: "I did not lock it. But I saw a woman with a seal on her sleeve. She said the bell was broken. I believed her. I should not have."
# DIRAC opt_04_2: ["name_on_clapper", "c5"]
ORX opt_04_3/rxn_04_3 -> enc_05 | O:Offer to walk the streets with him at night | D:C(1)
# R rxn_04_3: "Walk with me, then. If the silence is a message, we will hear it together. And if it is a theft, we will find the thief."
# DIRAC opt_04_3: ["plaza_echo", "c1"]

ENC enc_05 turn=0..999
# TITLE: The Archive of Names
# T: The Archive of Names smells of old paper and colder dust. Sable moves through the aisles with the ease of someone who has memorized every shelf, every drawer, every hidden gap between the ledgers. "Here," she says, pulling out a slim volume. "The city's records. The bell was never officially installed."
# T: 
# T: Mara frowns. "Never installed? Then what was ringing?"
# T: 
# T: Sable's smile is thin. "That is the question, isn't it? If the bell was never installed, then the silence is not a loss. It is a correction. The city has been ringing a lie for a hundred years."
ORX opt_05_1/rxn_05_1 -> enc_06 | O:Ask Sable who wrote the records | D:C(1)
# R rxn_05_1: "I did not write them. But I know who did. And I know they are no longer in this city."
# DIRAC opt_05_1: ["ledger_tear", "c2"]
ORX opt_05_2/rxn_05_2 -> enc_06 | O:Examine the ledger for hidden entries | D:C(1)
# R rxn_05_2: Sable watches Mara's fingers trace the page. "Careful. Some names are not meant to be read. Some silences are not meant to be broken."
# DIRAC opt_05_2: ["undercroft_bell", "c3"]
ORX opt_05_3/rxn_05_3 -> enc_06 | O:Ask what the city did with the bell before it was lost | D:C(1)
# R rxn_05_3: "We sold it. Not the bronze. The rhythm. The sound. We sold it to collectors who wanted to own a piece of the city's voice."
# DIRAC opt_05_3: ["market_trade", "c4"]

ENC enc_06 turn=0..999
# TITLE: The Torn Page
# T: Sable produces the page from her coat, the paper brittle and dark at the edges. "This is older than the tower," she says. "Older than the city. The bell was never installed because it was never meant to be a bell. It was meant to be a name."
# T: 
# T: Mara takes the page. The ink is faded but legible: a list of names, each one crossed out with a single, deliberate stroke. "These are not people," she whispers. "These are silences. Each one a sound the city chose to forget."
# T: 
# T: Sable's voice drops. "And the last name on the list... is yours."
ORX opt_06_1/rxn_06_1 -> enc_07 | O:Ask Sable how she knows the page is older than the tower | D:C(1)
# R rxn_06_1: "I have held it in my hands for longer than I can say. The paper remembers what the city has forgotten."
# DIRAC opt_06_1: ["ledger_tear", "c2"]
ORX opt_06_2/rxn_06_2 -> enc_07 | O:Examine the crossed-out names more closely | D:C(1)
# R rxn_06_2: Mara's breath catches. "These are not just silences. They are people. Each one erased from the Archive of Names."
# DIRAC opt_06_2: ["name_on_clapper", "c3"]
ORX opt_06_3/rxn_06_3 -> enc_07 | O:Ask what the city will do with the page now that it is found | D:C(1)
# R rxn_06_3: "Nothing. The city will pretend it was never here. As it has always done. As it always will."
# DIRAC opt_06_3: ["plaza_echo", "c5"]

ENC enc_07 turn=0..999
# TITLE: The Night Market
# T: Mara slips into the Night Market where merchants trade in 'quiet hours' as if silence were currency, and the air hums with hushed transactions. Stalls glow with lanterns that burn too low, and every voice is a whisper, as though the darkness itself is listening. A merchant in a grey cloak leans toward her. "You're looking for the bell's old rhythm? It's not lost. It's been sold, piece by piece, to people who can't stand the sound of their own names." He slides a small vial across the table. Inside, something dark and still moves like a held breath.
# T: 
# T: Mara watches the crowd. A woman in a red shawl buys a quiet hour from a man with no face, and the man's mouth moves but no sound comes out. The woman leaves lighter, as if a weight has been lifted from her shoulders. "What do you get back?" Mara asks. The merchant shrugs. "A memory. A name. A piece of the city that used to be yours. Silence is the only thing that never devalues."
# T: 
# T: She thinks of the wax seal on the tower door, the one that matched no known guild. She thinks of the torn page from the Archive of Names, dated before the city existed. The bell was never officially installed. It was never meant to be a bell at all. It was meant to be a name. And now someone is selling that name in pieces, like a merchant selling off the last of a dying fire.
# T: 
# T: The merchant's eyes catch the lantern light. "You're not here for the rhythm, are you? You're here for the truth. But truth is expensive in this market. And you can't pay with what you don't have."
ORX opt_07_1/rxn_07_1 -> enc_08 | O:Ask about the wax seal | D:C(1)
# R rxn_07_1: "The seal on the tower door. It's not a guild mark. It's someone's private hand. Who put it there?"
# DIRAC opt_07_1: ["seal_check", "c0"]
ORX opt_07_2/rxn_07_2 -> enc_08 | O:Buy a quiet hour | D:C(1)
# R rxn_07_2: "I'll trade a memory for a quiet hour. Tell me what silence costs in this market."
# DIRAC opt_07_2: ["market_trade", "c1"]
ORX opt_07_3/rxn_07_3 -> enc_08 | O:Follow the red shawl | D:C(1)
# R rxn_07_3: "That woman in red. She bought a quiet hour and left lighter. What did she give up?"
# DIRAC opt_07_3: ["plaza_echo", null]

ENC enc_08 turn=0..999
# TITLE: Tobias the Rhythm Seller
# T: She meets Tobias, who reveals he has been selling the bell's old rhythm to collectors, and his smile is as hollow as the tower. He sits in a small room above a shop that sells nothing but silence, in jars and vials and folded paper. The walls are lined with glass containers, each holding a different shade of quiet. Some are pale as morning fog. Others are dark as the space between stars.
# T: 
# T: "You want to know about the rhythm?" Tobias says, pouring two cups of tea that steams too slowly. "I didn't steal it. I lent it out. The city was drowning in noise, and I thought if I could take the bell's voice and spread it thin, maybe people would finally hear themselves. But collectors don't want to hear. They want to own. They want to keep the rhythm in a jar and play it when they feel like remembering what it was to be alive."
# T: 
# T: Mara watches his hands. They move with a strange precision, as if he's still counting something. "How many have you sold?" she asks. "Dozens. Maybe more. Each one a piece of the city's memory. Each one a name that will never be spoken again. The Archive of Names is emptying, Mara. Not because people are dying. Because they're being sold."
# T: 
# T: The tea cools between them. Outside, the market whispers continue, but here in this room, the silence has weight. It presses against her ears like deep water. "You could stop it," she says. "You could give it back. Let the city hear its own name again."
# T: 
# T: Tobias smiles, and for a moment his eyes are not hollow but bright, as if something behind them is trying to break through. "And who would I be, if I gave it back? A thief who returned what he stole? Or a man who finally admitted he was the one who locked the tower door?"
ORX opt_08_1/rxn_08_1 -> enc_09 | O:Demand the rhythm back | D:C(1)
# R rxn_08_1: "Give it back. Every piece. The city needs to hear its own name again."
# DIRAC opt_08_1: ["ledger_tear", "c2"]
ORX opt_08_2/rxn_08_2 -> enc_09 | O:Ask about the collectors | D:C(1)
# R rxn_08_2: "Who are the collectors? What do they want with the rhythm?"
# DIRAC opt_08_2: ["market_trade", "c4"]
ORX opt_08_3/rxn_08_3 -> enc_09 | O:Show him the torn page | D:C(1)
# R rxn_08_3: "I have a page from the Archive. Dated before the city existed. It says the bell was never officially installed."
# DIRAC opt_08_3: ["name_on_clapper", "c3"]

ENC enc_09 turn=0..999
# TITLE: The Borrowed Hours
# T: A merchant offers Mara a quiet hour in exchange for a memory, and she refuses, but the deal leaves a cold mark on her wrist. The merchant is a tall woman with silver hair and eyes that have seen too many transactions. She sits across from Mara in a booth that smells of old paper and colder dust. "I can give you an hour of silence," she says, sliding a small glass vial across the table. "One hour. No sound. No thought. No name. Just quiet. All I ask is a memory. Any memory. The city will take care of the rest."
# T: 
# T: Mara looks at the vial. Inside, the silence moves like a living thing, dark and patient. She thinks of the bell in the Silent Tower, locked from the inside. She thinks of the public notice forbidding repair. She thinks of the wax seal that matched no known guild, and the torn page that said the bell was never officially installed. She thinks of Tobias, selling the rhythm in pieces, and the collectors who wanted to own the city's voice.
# T: 
# T: "No," she says. "I won't trade a memory for silence."
# T: 
# T: The merchant's smile does not fade. "Everyone refuses at first. But the mark stays. The cold mark on your wrist. It's a reminder. A debt. And debts, in this market, always come due."
# T: 
# T: Mara stands. The booth is empty now, except for the vial and the cold mark on her wrist. Outside, the Night Market continues its hushed transactions, and the silence presses against her like deep water. She walks through the crowd, and for a moment she hears it: the bell's old rhythm, faint and broken, moving through the streets like a ghost trying to remember its own name.
ORX opt_09_1/rxn_09_1 -> enc_10 | O:Refuse the deal | D:C(1)
# R rxn_09_1: "No. I won't trade a memory for silence. The city's voice is not for sale."
# DIRAC opt_09_1: ["plaza_echo", "c5"]
ORX opt_09_2/rxn_09_2 -> enc_11 | O:Ask about the cold mark | D:C(1)
# R rxn_09_2: "What is this mark? What will it cost me when it comes due?"
# DIRAC opt_09_2: ["undercroft_bell", null]
ORX opt_09_3/rxn_09_3 -> enc_10 | O:Search the undercroft | D:C(1)
# R rxn_09_3: "Take me to the undercroft. I want to see the second bell. The one with my name on its clapper."
# DIRAC opt_09_3: ["name_on_clapper", "c3"]

ENC enc_10 turn=0..999
# TITLE: The Collector's Ledger
# T: Tobias unrolls a leather ledger across the counter, its pages swollen with ink and wax. "Every rhythm I have sold is here," he says, tapping a column of names. "Buyers from the north, from the coast, from places that no longer appear on any map. When they pay, I cross them out. Not because they are dead. Because they are no longer needed."
# T: 
# T: Mara leans closer. The crossed-out names are not merely erased; they are overwritten with smaller, more careful letters, as if someone had tried to make the people disappear twice. "You are not selling sound," she says. "You are selling absence."
# T: 
# T: Tobias smiles, but it does not reach his eyes. "Absence is the only thing that cannot be taxed. The city pays for noise. It pays for light. But silence? Silence is free. And I am simply the one who knows how to make it expensive."
ORX opt_10_1/rxn_10_1 -> enc_11 | O:Ask who crossed out the names | D:C(1)
# R rxn_10_1: "You did not cross them out yourself, did you? Someone else held the pen."
# DIRAC opt_10_1: ["seal_check", "c0"]
ORX opt_10_2/rxn_10_2 -> enc_11 | O:Demand the ledger be opened to the city | D:C(1)
# R rxn_10_2: "If this is a market for absence, then the city has a right to see what it has sold."
# DIRAC opt_10_2: ["market_trade", "c1"]
ORX opt_10_3/rxn_10_3 -> enc_11 | O:Tear out a page and keep it | D:C(1)
# R rxn_10_3: "I will not let you hide this in a locked room. If the city is buying silence, it should know what it is buying."
# DIRAC opt_10_3: ["ledger_tear", "c2"]

ENC enc_11 turn=0..999
# TITLE: The Unwritten Ledger
# T: The Archive of Names smells of dust and old wax. Mara runs her fingers along the spines of the ledgers, searching for the one that should contain the bell's installation. She finds it, but the page where the bell should be recorded is not merely blank; it is shaped like a mouth, a gap cut into the parchment as if someone had bitten through the record itself.
# T: 
# T: Sable stands behind her, arms crossed. "You see it now," she says. "The city did not lose its bell. It swallowed it. And the Archive is the throat."
# T: 
# T: Mara turns the ledger over. On the back, in handwriting she recognizes as her own, are the words: "The bell was never officially installed. It was a name given to a sound that had no right to exist."
# T: 
# T: "You wrote this," Sable says. It is not a question. "You have been here before. You have been here many times."
ORX opt_11_1/rxn_11_1 -> enc_12 | O:Ask Sable about the wax seal | D:C(1)
# R rxn_11_1: "That seal on the tower door. It matches no guild. But it looks like your hand. Explain."
# DIRAC opt_11_1: ["seal_check", "c0"]
ORX opt_11_2/rxn_11_2 -> enc_12 | O:Search for the second bell | D:C(1)
# R rxn_11_2: "You mentioned a second bell. A smaller one. Where is it? I need to see it before I can understand what this gap means."
# DIRAC opt_11_2: ["undercroft_bell", "c3"]
ORX opt_11_3/rxn_11_3 -> enc_12 | O:Ask about the public notice | D:C(1)
# R rxn_11_3: "The notice says the bell is broken. But you know it was silenced. Who put that notice up, and why did no one admit it?"
# DIRAC opt_11_3: ["plaza_echo", "c5"]

ENC enc_12 turn=0..999
# TITLE: The Page Before the City
# T: Sable holds the torn page up to the candlelight. The ink is old, the paper brittle, but the words are clear: a record of the bell's installation, dated before the city existed. "This is not a weapon," Sable says. "It is a mirror. If you look long enough, you will see the city eating itself."
# T: 
# T: Mara takes the page. The words blur and sharpen, blur and sharpen, until she sees not just the record but the shape of the city around it, the streets and towers and people all turning inward, consuming their own history.
# T: 
# T: "You tore this out," Mara says. "Before the city existed. You kept it hidden. Why?"
# T: 
# T: Sable's face is unreadable. "Because the city does not want to remember. And I am the one who remembers for it. But remembering is not the same as keeping. And keeping is not the same as owning."
ORX opt_12_1/rxn_12_1 -> enc_13 | O:Ask Sable to return the page | D:C(1)
# R rxn_12_1: "The city's voice is speaking from the undercroft. It is asking you to return the page. Accept that memory is not yours to burn."
# DIRAC opt_12_1: ["name_on_clapper", "c3"]
ORX opt_12_2/rxn_12_2 -> enc_13 | O:Ask about the collector's demand | D:C(1)
# R rxn_12_2: "Tobias said a collector demanded payment in names. Is that why you tore this page? To keep the city from being erased?"
# DIRAC opt_12_2: ["market_trade", "c4"]
ORX opt_12_3/rxn_12_3 -> enc_13 | O:Ask who silenced the bell | D:C(1)
# R rxn_12_3: "The notice says the bell is broken. But you know it was silenced. Who did it, and why? I need to know before I decide what to do with this page."
# DIRAC opt_12_3: ["plaza_echo", "c5"]

ENC enc_13 turn=0..999
# TITLE: The Undercroft
# T: Mara descends into the undercroft and finds a hidden chamber holding a second bell, smaller and older than the one above. The air is thick with dust and the faint smell of wax. She runs her fingers along the spines of the ledgers, searching for the one that should contain the bell's installation. She finds it, but the page is torn, and the words are gone.
# T: 
# T: Sable stands in the shadows, holding a candle. "You found it," she says. "The second bell. The one that was never meant to be here."
# T: 
# T: Mara looks at the bell. It is smaller than the one above, older, and covered in a thin layer of dust. "What is this?" she asks.
# T: 
# T: Sable steps forward. "It is the bell that was never officially installed. The one that was silenced before the city existed."
ORX opt_13_1/rxn_13_1 -> enc_14 | O:Examine the wax seal on the tower door | D:C(1)
# R rxn_13_1: Mara: "This seal... it matches no known guild. It's an unauthorized entry."
# DIRAC opt_13_1: ["seal_check", "c0"]
ORX opt_13_2/rxn_13_2 -> enc_14 | O:Ask Sable about the quiet hours market | D:C(1)
# R rxn_13_2: Sable: "Merchants trade in silence as if it were currency. The market for absence is real."
# DIRAC opt_13_2: ["market_trade", "c1"]
ORX opt_13_3/rxn_13_3 -> enc_14 | O:Search the chamber for more clues | D:C(1)
# R rxn_13_3: Mara: "There's a page torn from the Archive of Names ledger here. Dated before the city existed."
# DIRAC opt_13_3: ["ledger_tear", "c2"]

ENC enc_14 turn=0..999
# TITLE: Her Name on the Clapper
# T: She hears her own name inscribed on the second bell's clapper, and the sound of her heartbeat echoes in the stone. The chamber is silent except for the faint ticking of her pulse. She looks at the bell, and the words are clear: a record of the bell's installation, dated before the city existed.
# T: 
# T: Sable watches from the shadows. "You see it now," she says. "Your name. On the clapper. You were here before the city existed."
# T: 
# T: Mara touches the bell. It is warm, as if it has been waiting for her. "What does this mean?" she asks.
# T: 
# T: Sable steps closer. "It means you are not just a witness. You are part of the story. The bell was never officially installed because it was never meant to be found."
ORX opt_14_1/rxn_14_1 -> enc_15 | O:Strike the small bell | D:C(1)
# R rxn_14_1: Mara: "The sound is different. It's not quite a note, but something older."
# DIRAC opt_14_1: ["undercroft_bell", "c3"]
ORX opt_14_2/rxn_14_2 -> enc_15 | O:Ask Sable about the collector's demand | D:C(1)
# R rxn_14_2: Sable: "A collector demands payment in names. His quiet-hours trade has been erasing people from the Archive of Names."
# DIRAC opt_14_2: ["name_on_clapper", "c4"]
ORX opt_14_3/rxn_14_3 -> enc_15 | O:Look for the public notice | D:C(1)
# R rxn_14_3: Mara: "The public notice claims the bell is broken. But no one admits it was actually silenced."
# DIRAC opt_14_3: ["plaza_echo", "c5"]

ENC enc_15 turn=0..999
# TITLE: The Counter-Sound
# T: Mara strikes the small bell and hears a note that is not quite a note, but something older, like a word before it became a word. The sound fills the chamber, and the dust motes dance in the candlelight. She looks at Sable, who is watching with a mixture of fear and hope.
# T: 
# T: "What was that?" Mara asks.
# T: 
# T: Sable's voice is quiet. "It is the counter-sound. The sound that was silenced before the city existed. It is the sound of the bell that was never officially installed."
# T: 
# T: Mara looks at the bell again. Her name is still there, on the clapper. She thinks of the wax seal, the quiet hours market, the torn page from the Archive of Names. She thinks of Tobias selling the bell's old rhythm to collectors. She thinks of the public notice claiming the bell is broken.
# T: 
# T: "What do we do now?" she asks.
ORX opt_15_1/rxn_15_1 -> enc_16 | O:Return the page to the Archive | D:C(1)
# R rxn_15_1: Mara: "The page is not a weapon; it is a mirror. If you look long enough, you will see the city eating itself."
# DIRAC opt_15_1: ["ledger_tear", "c2"]
ORX opt_15_2/rxn_15_2 -> enc_16 | O:Sell the rhythm to the collectors | D:C(1)
# R rxn_15_2: Sable: "I only lend the rhythm out; I never meant to keep it. The city should hear it again in a new shape."
# DIRAC opt_15_2: ["market_trade", "c4"]
ORX opt_15_3/rxn_15_3 -> enc_17 | O:Silence the bell forever | D:C(1)
# R rxn_15_3: Mara: "The silence is not a theft, a mercy, or a message. It is a choice. And we must make it together."
# DIRAC opt_15_3: ["undercroft_bell", "c3"]

ENC enc_16 turn=0..999
# TITLE: The City's Hunger
# T: Sable stands before the torn page, her fingers tracing the ink that predates the city's first stone. "You think the bell was silenced? No. The city ate it. Every year it swallows a memory, and the silence is just the chewing." She holds the page up to the candlelight, and the letters seem to move, like something alive beneath the parchment.
# T: 
# T: Mara watches the page tremble in Sable's hand. "If the city eats history, then what are we? Just another meal waiting to be swallowed?" Tobias steps forward, his voice low. "I've seen the collectors. They don't buy silence for its own sake. They buy it because it's the only thing the city can't digest."
# T: 
# T: Sable's eyes flick between them. "The Archive of Names isn't a record. It's a menu. And we're all still on it." The candle gutters, and for a moment the chamber is dark enough to hear the city breathing outside.
ORX opt_16_1/rxn_16_1 -> enc_17 | O:Ask Sable to show the ledger page | D:C(1)
# R rxn_16_1: Sable: "You want to see what the city ate? Fine. But once you read it, you can't unsee it. The names are still warm."
# DIRAC opt_16_1: ["seal_check", null]
ORX opt_16_2/rxn_16_2 -> enc_17 | O:Demand Tobias reveal his collectors | D:C(1)
# R rxn_16_2: Tobias: "They're not collectors. They're archivists of what's been lost. And I'm the only one who remembers how to read them."
# DIRAC opt_16_2: ["market_trade", "c4"]
ORX opt_16_3/rxn_16_3 -> enc_17 | O:Search the tower door for the seal | D:C(1)
# R rxn_16_3: Mara: "If the city ate the bell, it left a mark. That wax seal on the door—it's not a guild. It's a tooth mark."
# DIRAC opt_16_3: ["seal_check", "c0"]

ENC enc_17 turn=0..999
# TITLE: The Plaza of Echoes
# T: The Plaza of Echoes is a shallow bowl of stone, and every word spoken here returns to the speaker slightly altered. Mara steps into the center and says, "We need to decide what to restore." The echo comes back as, "We need to decide what to restore... or what to let die."
# T: 
# T: Tobias laughs, but the echo makes it sound like a sob. "I've been selling the rhythm because someone had to. If I kept it, the city would have eaten me too. Now it's in their hands, and they're using it to buy silence from each other." He pulls a small vial from his coat. "This is the last of the old rhythm. One drop, and you'll hear what the bell sounded like before it was swallowed."
# T: 
# T: Sable watches them both, her hand on the torn page. "You can't restore what the city has already digested. But you can make it hungry again. You can make it remember it's supposed to eat." The plaza holds its breath, and the echoes begin to layer, overlapping into something almost like a chorus.
ORX opt_17_1/rxn_17_1 -> enc_18 | O:Let Tobias play the last rhythm | D:C(1)
# R rxn_17_1: Tobias: "Listen. That's not a note. That's a word. The city was never silent—it was just holding its breath."
# DIRAC opt_17_1: ["plaza_echo", "c4"]
ORX opt_17_2/rxn_17_2 -> enc_18 | O:Ask Sable to read the page aloud | D:C(1)
# R rxn_17_2: Sable: "If I read it here, the plaza will echo it back. And the city will hear itself being named. That's not a choice. That's a confession."
# DIRAC opt_17_2: ["name_on_clapper", "c2"]
ORX opt_17_3/rxn_17_3 -> enc_18 | O:Strike the small bell in the plaza | D:C(1)
# R rxn_17_3: Mara: "The second bell. The one with my name on it. If I strike it here, the echo will carry it into every street. The city can't eat what it hears."
# DIRAC opt_17_3: ["undercroft_bell", "c3"]

ENC enc_18 turn=0..999
# TITLE: The Choice of Sound
# T: The plaza is silent now, the echoes fading into the stone. Three choices hang in the air, and each one will change what Vespera becomes. The tower door stands open behind them, the wax seal still fresh on the wood.
# T: 
# T: Tobias holds out the vial. "If we restore the old rhythm, the city will remember it was ever a bell at all. But remembering means hunger. It'll eat again. Faster this time." His voice is steady, but his hands are not.
# T: 
# T: Sable steps forward, the torn page in her hand. "Or we can let the silence stand. Let the city keep its meal. But then the Archive of Names stays a menu, and we're all still on it. I can feel the names being written out. One by one. Quietly."
# T: 
# T: Mara looks at the small bell in her palm, her name still warm on the clapper. "Or we can make a new sound. Not the old rhythm, not the silence. Something the city has never eaten before. Something it can't digest." She holds the bell up, and the plaza waits.
ORX opt_18_1/rxn_18_1 -> enc_19 | O:Restore the old rhythm | D:C(1)
# R rxn_18_1: Tobias: "Then let it ring. Let the city remember what it was. And let it be hungry again. I'd rather it be hungry than full of lies."
# DIRAC opt_18_1: ["plaza_echo", "c4"]
ORX opt_18_2/rxn_18_2 -> enc_19 | O:Let the silence stand | D:C(1)
# R rxn_18_2: Sable: "If we do this, I have to burn the page. The city will forget it ever had a bell. And I'll forget I ever had a name. But at least the menu will be empty."
# DIRAC opt_18_2: ["ledger_tear", "c2"]
ORX opt_18_3/rxn_18_3 -> enc_19 | O:Make a new sound | D:C(1)
# R rxn_18_3: Mara: "Not the old rhythm. Not silence. Something new. The city can't eat what it hasn't tasted yet. And if we're going to be on that menu, let's be the dish it never expected."
# DIRAC opt_18_3: ["undercroft_bell", "c3"]

ENC enc_19 turn=0..999
# TITLE: The True Culprit
# T: The undercroft door groans shut behind them, and the air tastes of wet stone and old ink. Mara presses her palm to the wall and feels it pulse, slow as a sleeping animal. "It was never a thief," she says, her voice thin in the dark. "It was never a guild, or a merchant, or even me. The city ate the bell. Every year it swallows a memory, and the silence is just the chewing."
# T: 
# T: Sable steps forward, the torn page trembling in her hand. "You think I silenced it? I only copied the seal so no one could use it against me. But look at this ink. It predates the first stone. The bell was never officially installed. It was a mouth, and the city has been feeding it its own past."
# T: 
# T: Tobias leans against the pillar, his collector's satchel clinking softly. "I only lend the rhythm out; I never meant to keep it. But the city should hear it again in a new shape. The silence is not a theft. It is a mercy. A message. And if you listen, you will hear the city speaking back."
# T: 
# T: The small bell hangs in the center of the chamber, its clapper still, its bronze skin dark with age. Mara's name is inscribed on it in letters so fine they might be scratches. The air hums, and for a moment the three of them are not alone.
ORX opt_19_1/rxn_19_1 -> enc_20 | O:Ask Sable to show the ledger page | D:C(1)
# R rxn_19_1: Sable: "The page is not a weapon; it is a mirror, and if you look long enough you will see the city eating itself."
# DIRAC opt_19_1: ["name_on_clapper", "c2"]
ORX opt_19_2/rxn_19_2 -> enc_20 | O:Press Mara to the tower door | D:C(1)
# R rxn_19_2: Mara: "The seal is not a guild mark; it is my own hand, and I have been the one who locked the tower door."
# DIRAC opt_19_2: ["seal_check", "c0"]
ORX opt_19_3/rxn_19_3 -> enc_20 | O:Demand Tobias's collector's ledger | D:C(1)
# R rxn_19_3: Tobias: "I only lend the rhythm out; I never meant to keep it, and the city should hear it again in a new shape."
# DIRAC opt_19_3: ["market_trade", "c4"]

ENC enc_20 turn=0..999
# TITLE: The Bell's Memory
# T: Mara strikes the small bell with a stone, and the note that rises is not quite a note. It is something older, like a word before it became a word. The sound does not echo; it settles into the floor, into the walls, into the bones of the city above.
# T: 
# T: "It was never broken," Mara whispers, her eyes wide. "It was a mouth. The city has been feeding it its own past, and the silence is just the chewing."
# T: 
# T: Sable kneels beside the bell, tracing the inscription with one finger. "The public notice claimed the bell was broken. No one admits it was silenced. But this bell was never officially installed. It was a mouth, and the city has been feeding it its own past."
# T: 
# T: Tobias watches from the doorway, his satchel clinking softly. "I only lend the rhythm out; I never meant to keep it. But the city should hear it again in a new shape. The silence is not a theft. It is a mercy. A message. And if you listen, you will hear the city speaking back."
ORX opt_20_1/rxn_20_1 -> enc_21 | O:Strike the bell again | D:C(1)
# R rxn_20_1: Mara: "It was never broken; it was a mouth, and the city has been feeding it its own past."
# DIRAC opt_20_1: ["undercroft_bell", "c3"]
ORX opt_20_2/rxn_20_2 -> enc_21 | O:Read the public notice aloud | D:C(1)
# R rxn_20_2: Sable: "The public notice claimed the bell was broken, which no one admits was actually silenced."
# DIRAC opt_20_2: ["plaza_echo", "c5"]
ORX opt_20_3/rxn_20_3 -> enc_21 | O:Trade a memory for the rhythm | D:C(1)
# R rxn_20_3: Tobias: "I only lend the rhythm out; I never meant to keep it, and the city should hear it again in a new shape."
# DIRAC opt_20_3: ["market_trade", "c4"]

ENC enc_21 turn=0..999
# TITLE: The Second Bell's Song
# T: The small bell begins to ring on its own. Its song is not a melody; it is the city's name, spoken in a voice no one remembers. The sound fills the undercroft, presses against the walls, and for a moment the stone above them seems to breathe.
# T: 
# T: Mara's hand flies to her mouth. "That is my name. That is the city's name. That is the name of every person who has ever lived here and been forgotten."
# T: 
# T: Sable grips the torn page, her knuckles white. "The city's own voice speaking from the undercroft, asking her to return the page and accept that memory is not hers to burn."
# T: 
# T: Tobias steps forward, his collector's satchel clinking softly. "I only lend the rhythm out; I never meant to keep it. But the city should hear it again in a new shape. The silence is not a theft. It is a mercy. A message. And if you listen, you will hear the city speaking back."
ORX opt_21_1/rxn_21_1 -> enc_22 | O:Let the bell ring | D:C(1)
# R rxn_21_1: Mara: "That is my name. That is the city's name. That is the name of every person who has ever lived here and been forgotten."
# DIRAC opt_21_1: ["undercroft_bell", "c3"]
ORX opt_21_2/rxn_21_2 -> enc_22 | O:Return the ledger page | D:C(1)
# R rxn_21_2: Sable: "The city's own voice speaking from the undercroft, asking her to return the page and accept that memory is not hers to burn."
# DIRAC opt_21_2: ["ledger_tear", "c2"]
ORX opt_21_3/rxn_21_3 -> enc_23 | O:Silence the bell | D:C(1)
# R rxn_21_3: Tobias: "I only lend the rhythm out; I never meant to keep it, and the city should hear it again in a new shape."
# DIRAC opt_21_3: ["plaza_echo", "c5"]

ENC enc_22 turn=0..999
# TITLE: The Archive's Confession
# T: The Archive of Names opens its doors and spills out every record it has ever hidden, and the pages are warm, like skin. They drift across the floor in a slow tide, their edges curling as if breathing. Mara watches them settle around her feet, each one carrying the faint smell of ink and old rain.
# T: 
# T: Sable kneels among the pages, her fingers trembling as she touches a ledger that has never been opened before. "You think the bell was silenced? No. The city ate it. Every year it swallows a memory, and the silence is just the chewing." Her voice is low, almost a whisper, but the words hang in the air like smoke.
# T: 
# T: Tobias stands at the edge of the room, his collector's satchel clutched tight against his chest. He watches the pages with the wary expression of a man who has seen too many secrets and not enough coin. "I only lend the rhythm out; I never meant to keep it, and the city should hear it again in a new shape," he says, though his eyes dart toward the door as if expecting collectors to arrive at any moment.
ORX opt_22_1/rxn_22_1 -> enc_23 | O:Ask Sable about the wax seal | D:C(1)
# R rxn_22_1: Sable: "The seal is not a guild mark; it is my own hand, and I have been the one who locked the tower door. I kept it as a private record of who silenced it."
# DIRAC opt_22_1: ["seal_check", "c0"]
ORX opt_22_2/rxn_22_2 -> enc_23 | O:Press Tobias on the quiet-hours trade | D:C(1)
# R rxn_22_2: Tobias: "Merchants trading in 'quiet hours' as if silence were currency, revealing a market for absence. I sell the rhythm to collectors, but I never meant to keep it."
# DIRAC opt_22_2: ["market_trade", "c1"]
ORX opt_22_3/rxn_22_3 -> enc_23 | O:Demand the torn ledger page | D:C(1)
# R rxn_22_3: Sable: "A page torn from the Archive of Names ledger, dated before the city existed, showing the bell was never officially installed. The page is not a weapon; it is a mirror, and if you look long enough you will see the city eating itself."
# DIRAC opt_22_3: ["ledger_tear", "c2"]

ENC enc_23 turn=0..999
# TITLE: Tobias's Last Rhythm
# T: Tobias offers to give back the rhythm he sold, but the collectors have already spent it, and the silence is now in their bones. He stands in the center of the undercroft, his satchel open, revealing a small brass bell that has never been rung. The air around him hums with a faint, almost imperceptible vibration.
# T: 
# T: "I only lend the rhythm out; I never meant to keep it," Tobias says, his voice cracking as he speaks. "The city should hear it again in a new shape. But the collectors have already spent it. They bought the silence and wore it like a cloak. Now it is in their bones, and I cannot take it back."
# T: 
# T: Mara watches him with a mixture of pity and suspicion. She has seen the way he moves through the city, always listening, always selling. But she also knows that the rhythm he sold was not just a sound; it was a memory, a piece of the city's soul. And now it is gone, scattered among the collectors who have no intention of returning it.
ORX opt_23_1/rxn_23_1 -> enc_24 | O:Ask Tobias to sell the rhythm back | D:C(1)
# R rxn_23_1: Tobias: "I can sell the rhythm back, but the collectors have already spent it. The silence is now in their bones, and I cannot take it back."
# DIRAC opt_23_1: ["market_trade", "c4"]
ORX opt_23_2/rxn_23_2 -> enc_24 | O:Demand the collector's name | D:C(1)
# R rxn_23_2: Tobias: "A collector demanding payment in names, revealing that his quiet-hours trade has been erasing people from the Archive of Names. I never meant to keep it, and the city should hear it again in a new shape."
# DIRAC opt_23_2: ["name_on_clapper", "c5"]
ORX opt_23_3/rxn_23_3 -> enc_24 | O:Strike the small bell | D:C(1)
# R rxn_23_3: Tobias: "The small bell begins to ring on its own. Its song is not a melody; it is the city's name, spoken in a voice no one remembers. The sound fills the undercroft, presses against the walls."
# DIRAC opt_23_3: ["undercroft_bell", "c3"]

ENC enc_24 turn=0..999
# TITLE: Sable's Page
# T: Sable returns the torn page to the Archive, and the city shudders, as if a tooth has been pulled from its jaw. The pages around her begin to glow with a faint, golden light, their edges curling and uncurling like living things. She holds the page in both hands, her fingers tracing the ink that predates the city's first stone.
# T: 
# T: "The page is not a weapon; it is a mirror," Sable says, her voice steady despite the trembling of her hands. "If you look long enough you will see the city eating itself. The bell was never officially installed. It was a memory, a dream, a lie the city told itself to keep from going mad."
# T: 
# T: Mara watches as Sable places the page back into the Archive. The moment it touches the shelf, the pages around it begin to move, rearranging themselves, forming new patterns. The air grows thick with the smell of ink and old rain, and the sound of the city's heartbeat becomes louder, more insistent.
ORX opt_24_1/rxn_24_1 -> enc_25 | O:Ask Sable about the public notice | D:C(1)
# R rxn_24_1: Sable: "The public notice claiming the bell is broken, which no one admits was actually silenced. I tore the ledger page before the city existed and kept it hidden. The city's own voice speaking from the undercroft, asking her to return the page and accept that memory is not hers to burn."
# DIRAC opt_24_1: ["plaza_echo", "c5"]
ORX opt_24_2/rxn_24_2 -> enc_25 | O:Demand Sable confess | D:C(1)
# R rxn_24_2: Sable: "The seal is not a guild mark; it is my own hand, and I have been the one who locked the tower door. I kept it as a private record of who silenced it. A public ledger showing her name beside the second bell's clapper, forcing her to choose between order and confession."
# DIRAC opt_24_2: ["seal_check", "c0"]
ORX opt_24_3/rxn_24_3 -> enc_25 | O:Strike the small bell again | D:C(1)
# R rxn_24_3: Sable: "The small bell begins to ring on its own. Its song is not a melody; it is the city's name, spoken in a voice no one remembers. The sound fills the undercroft, presses against the walls."
# DIRAC opt_24_3: ["undercroft_bell", "c3"]

ENC enc_25 turn=0..999
# TITLE: The Bell Tower Reopens
# T: Mara presses the wax seal against the tower door, and the lock turns with a sound like a held breath finally released. The great bell is gone, replaced by a hollow space shaped like a bell, its absence carved into the stone. She stands in the empty frame, her fingers tracing the outline of what was there, and feels the city's silence pressing against her ribs.
# T: 
# T: Tobias watches from the doorway, his satchel slung low, and says nothing. He has sold the rhythm to collectors, but the shape of the bell remains in his memory, and he cannot stop seeing it. Sable steps forward, her hand on the torn page she has returned, and asks if the hollow is a wound or a mouth.
# T: 
# T: Mara answers that it is both, and that the city has been eating its own history. The three of them stand in the empty tower, and the silence between them is not empty but full, like a bell that has been struck and is still ringing in the dark.
ORX opt_25_1/rxn_25_1 -> enc_26 | O:Press the seal deeper into the hollow | D:C(1)
# R rxn_25_1: Mara: 'If the city took the bell, it will take the seal too. I will not let it keep my hand.'
# DIRAC opt_25_1: ["seal_check", "c0"]
ORX opt_25_2/rxn_25_2 -> enc_26 | O:Ask Tobias what the collectors paid | D:C(1)
# R rxn_25_2: Tobias: 'They paid in names. Not yours. Not mine. But close enough to hurt.'
# DIRAC opt_25_2: ["market_trade", "c4"]
ORX opt_25_3/rxn_25_3 -> enc_26 | O:Let Sable read the page aloud | D:C(1)
# R rxn_25_3: Sable: 'The city is not broken. It is listening. And it has been listening for a very long time.'
# DIRAC opt_25_3: ["ledger_tear", "c2"]

ENC enc_26 turn=0..999
# TITLE: The Echo Returns
# T: In the Plaza of Echoes, the stones are warm and the air is thick with voices that are not echoes. They are the voices of everyone the city has swallowed, speaking in unison and in fragments, their words overlapping like a bell struck too many times. Mara hears her own name in the crowd of voices, spoken by someone who is not there.
# T: 
# T: Tobias kneels and presses his palm to the ground, and the rhythm he sold comes back to him, not as sound but as a pulse in the stone. He says the city has been eating its own history, and that the silence was not a theft but a digestion. Sable stands at the center of the plaza, her hand on the small bell with her name inscribed on its clapper, and asks if the city will speak if she rings it.
# T: 
# T: The voices in the plaza answer before she can finish. They say the city is not a thief. It is a mouth. And it is still hungry.
ORX opt_26_1/rxn_26_1 -> enc_27 | O:Ring the small bell | D:C(1)
# R rxn_26_1: Sable: 'If the city is a mouth, then this is the only thing it cannot swallow. My name. My voice.'
# DIRAC opt_26_1: ["undercroft_bell", "c3"]
ORX opt_26_2/rxn_26_2 -> enc_27 | O:Ask the echoes for the truth | D:C(1)
# R rxn_26_2: Echo: 'The truth is not a thing you can ask. It is a thing you must become. And you are not ready.'
# DIRAC opt_26_2: ["plaza_echo", "c5"]
ORX opt_26_3/rxn_26_3 -> enc_27 | O:Offer the wax seal to the voices | D:C(1)
# R rxn_26_3: Mara: 'I have been copying this seal for months. I thought it was a record. It is a key. And I have been holding it too long.'
# DIRAC opt_26_3: ["name_on_clapper", "c0"]

ENC enc_27 turn=0..999
# TITLE: The City's Voice
# T: The city's voice speaks from the undercroft, and it is not a voice at all but a pressure in the air, a weight in the bones. It asks Sable to return the page and accept that memory is not hers to burn. The three of them stand in the dark, and the small bell in Sable's hand grows warm, then hot, then almost too heavy to hold.
# T: 
# T: Mara says the city has been eating its own history, and that the silence was a mercy. Tobias says the rhythm is not his to give back, but the city can take it if it wants. Sable says nothing, and the city's voice waits, patient as stone, hungry as a bell that has been struck and not yet rung.
# T: 
# T: The page in Sable's hand glows with a faint golden light, and the words on it shift, rearrange, become something new. The city is not asking for the page. It is asking for the choice. And the choice is not hers alone.
ORX opt_27_1/rxn_27_1 -> enc_28 | O:Return the page to the Archive | D:C(1)
# R rxn_27_1: Sable: 'I will give it back. But I will not forget. And neither will you.'
# DIRAC opt_27_1: ["ledger_tear", "c2"]
ORX opt_27_2/rxn_27_2 -> enc_29 | O:Ring the small bell one last time | D:C(1)
# R rxn_27_2: Sable: 'If the city is a mouth, then this is the only thing it cannot swallow. My name. My voice. My choice.'
# DIRAC opt_27_2: ["undercroft_bell", "c3"]
ORX opt_27_3/rxn_27_3 -> enc_28 | O:Let the city keep the silence | D:C(1)
# R rxn_27_3: Mara: 'Perhaps the silence was not a theft. Perhaps it was a mercy. And perhaps we have been too loud for too long.'
# DIRAC opt_27_3: ["plaza_echo", "c5"]

ENC enc_28 turn=0..999
# TITLE: The Reconciliation
# T: Mara, Tobias, and Sable stand together in the Plaza of Echoes as the city offers them a choice: restore the bell and let it eat again, or let the silence remain and let the city starve.
# T: 
# T: Mara's hand hovers over the wax seal in her pocket. "I locked the tower door myself," she says, her voice steady but her fingers trembling. "I thought if I kept the bell silent, I could keep the city from swallowing itself. But I was wrong. The silence isn't mercy. It's a wound."
# T: 
# T: Tobias steps forward, his collector's satchel clinking with the old rhythm he's been selling. "I've been trading in absence," he admits, "as if silence were something you could bottle and sell. But the city doesn't want my coin. It wants its voice back. Even if that voice is broken. Even if it's a new one."
# T: 
# T: Sable holds up the torn page from the Archive of Names. "I tore it before the city existed," she says, "and I kept it hidden. I thought I was protecting the truth. But the truth isn't a weapon. It's a mirror. And if you look long enough, you'll see the city eating itself."
# T: 
# T: The plaza stones hum beneath their feet. The city is waiting. The choice is theirs: restore the bell and let it eat again, or let the silence remain and let the city starve.
ORX opt_28_1/rxn_28_1 -> enc_29 | O:Restore the Bell | D:C(1)
# R rxn_28_1: Mara: "Then we ring it. We let the city hear itself again, even if it hurts. Even if it eats."
# DIRAC opt_28_1: ["seal_check", "c0"]
ORX opt_28_2/rxn_28_2 -> enc_29 | O:Let Silence Remain | D:C(1)
# R rxn_28_2: Tobias: "Maybe the city doesn't need to eat. Maybe it just needs to listen. Maybe silence is its own kind of voice."
# DIRAC opt_28_2: ["undercroft_bell", "c5"]
ORX opt_28_3/rxn_28_3 -> enc_29 | O:A Third Path | D:C(1)
# R rxn_28_3: Sable: "What if we don't restore the old bell? What if we make a new sound? A sound that belongs to no one and everyone?"
# DIRAC opt_28_3: ["plaza_echo", "c3"]

ENC enc_29 turn=0..999
# TITLE: The New Sound
# T: They choose a third option: they ring the small bell, and its sound is not the old rhythm but a new one, a sound that belongs to no one and everyone.
# T: 
# T: Sable lifts the small bell from the undercroft. It is lighter than she expected, almost weightless, as if it has been waiting for this moment. The name on its clapper catches the light: her own name, written in ink that seems to shift and breathe.
# T: 
# T: "This bell was never meant to be rung alone," she says, her voice barely above a whisper. "It was meant to be rung with the city. With all of us."
# T: 
# T: Mara steps forward and places her hand on the bell's rim. "I locked the tower door to protect you," she says to Tobias, "but I was wrong. The city doesn't need protection from its own voice. It needs us to listen."
# T: 
# T: Tobias opens his satchel and takes out the old rhythm, now a small glass vial filled with something that looks like liquid sound. "I've been selling this for years," he says, "as if it were a commodity. But it's not. It's a memory. And memories aren't meant to be sold. They're meant to be shared."
# T: 
# T: He pours the vial into the small bell. The sound that emerges is not the old rhythm. It is something new. Something that belongs to no one and everyone. The plaza stones vibrate with it, and the voices of the swallowed rise to meet it, not in unison but in harmony.
ORX opt_29_1/rxn_29_1 -> enc_30 | O:Ring the Small Bell | D:C(1)
# R rxn_29_1: Sable: "This bell was never meant to be rung alone. It was meant to be rung with the city. With all of us."
# DIRAC opt_29_1: ["name_on_clapper", "c3"]
ORX opt_29_2/rxn_29_2 -> enc_30 | O:Share the Old Rhythm | D:C(1)
# R rxn_29_2: Tobias: "I've been selling this for years, as if it were a commodity. But it's not. It's a memory. And memories aren't meant to be sold. They're meant to be shared."
# DIRAC opt_29_2: ["market_trade", "c4"]
ORX opt_29_3/rxn_29_3 -> enc_30 | O:Let the City Listen | D:C(1)
# R rxn_29_3: Mara: "The city doesn't need protection from its own voice. It needs us to listen."
# DIRAC opt_29_3: ["plaza_echo", "c5"]

ENC enc_30 turn=0..999
# TITLE: The City That Lost Its Bell
# T: The city wakes with a new silence, not of absence but of listening, and Mara leaves the tower with the wax seal in her pocket, knowing she has been the one who locked the door all along.
# T: 
# T: The plaza is quiet, but it is not empty. The stones are warm beneath their feet, and the air is thick with voices that are not echoes. They are the voices of everyone the city has swallowed, speaking in unison and in fragments, and they are no longer angry. They are listening.
# T: 
# T: Mara stands at the tower door, the wax seal in her pocket. She knows now that the seal is not a guild mark. It is her own hand. She has been the one who locked the tower door, who silenced the bell, who kept the city from hearing itself. And yet, as she looks out over the plaza, she sees that the city is not broken. It is waiting. It is listening.
# T: 
# T: Tobias stands beside her, his satchel empty. "I sold the rhythm for years," he says, "as if it were something I could own. But it was never mine. It belonged to the city. To all of us."
# T: 
# T: Sable holds the torn page in her hand. "I tore it before the city existed," she says, "and I kept it hidden. I thought I was protecting the truth. But the truth isn't a weapon. It's a mirror. And if you look long enough, you'll see the city eating itself."
# T: 
# T: Mara places the wax seal on the tower door. The door opens. The city breathes. And in the new silence, not of absence but of listening, they all hear the small bell ringing again, its sound a new one, a sound that belongs to no one and everyone.
ORX opt_30_1/rxn_30_1 -> enc_finale | O:Leave the Tower | D:C(1)
# R rxn_30_1: Mara: "I locked the tower door myself. I thought I was protecting the city. But I was wrong. The city doesn't need protection from its own voice. It needs us to listen."
# DIRAC opt_30_1: ["seal_check", "c0"]
ORX opt_30_2/rxn_30_2 -> enc_finale | O:Return the Page | D:C(1)
# R rxn_30_2: Sable: "The page is not a weapon. It's a mirror. And if you look long enough, you'll see the city eating itself. I'm returning it to the Archive."
# DIRAC opt_30_2: ["ledger_tear", "c2"]
ORX opt_30_3/rxn_30_3 -> enc_finale | O:Accept the Silence | D:C(1)
# R rxn_30_3: Tobias: "The silence isn't absence. It's listening. And in that listening, the city is not broken. It's waiting. It's alive."
# DIRAC opt_30_3: ["undercroft_bell", "c5"]

ENC enc_finale turn=0..999
# TITLE: Act V: The Reconciliation
ORX opt_finale_1/rxn_finale_1 -> end_restore | O:The Bell Remembers Its Shape | D:C(1)
ORX opt_finale_2/rxn_finale_2 -> end_silence | O:The Quiet That Remains | D:C(1)
ORX opt_finale_3/rxn_finale_3 -> end_expose | O:The Name on the Clapper | D:C(1)
ORX opt_finale_4/rxn_finale_4 -> end_mix | O:The Sound Between | D:C(1)
# SECRET opt_finale_4: {"requires": ["c0", "c2", "c3"], "projectors": [["X0+", 0.45, 0.75], ["Y1+", 0.35, 0.6]]}
ORX opt_finale_5/rxn_finale_5 -> end_phase | O:The Silence That Was a Phase | D:C(1)
# SECRET opt_finale_5: {"requires": ["c1", "c2", "c5"], "projectors": [["X1-", 0.4, 0.7], ["Y0+", 0.3, 0.55]]}

ENC end_restore turn=0..999
# TITLE: The Bell Remembers Its Shape
# T: Mara pressed her thumb to the wax seal and felt the old order settle back into place, like a bone resetting after a fall. The tower door opened without protest, and the bell above rang once, clean and bright, as if it had only been sleeping. The city below exhaled, and the market stalls closed early, and no one asked why the silence had ever mattered at all.
# T: Sable watched from the plaza with her arms folded, and Tobias counted his coins with a face that said he had won something he could not name. The bell rang again, and again, and the city forgot why it was ever silent, and the forgetting felt like mercy, and the mercy felt like a lie, and no one in the tower would say which of those things was true.

ENC end_silence turn=0..999
# TITLE: The Quiet That Remains
# T: Tobias set the last coin on the counter and let the rhythm go, and the city listened to the absence the way a body listens to a wound that has stopped bleeding. The market closed early, and the quiet hours stretched out like a long breath held in a dark room, and people began to trade in silence the way they once traded in bread, and the Archive of Names grew thinner by one name each night.
# T: Mara stood in the plaza and felt the quiet settle into her bones, and she understood that the city had not lost its voice but had learned to speak in a language of absence, and the new ritual formed around the silence like moss on a stone, soft and patient and impossible to remove. She did not ring the bell again, and the city did not ask her to, and the quiet held them all like a hand over a mouth, gentle and absolute and full of things that would never be said aloud.

ENC end_expose turn=0..999
# TITLE: The Name on the Clapper
# T: Sable stood before the torn page and let the ink dry on her fingers, and the city's own voice rose from the undercroft like a thing that had been waiting a long time to be heard. The name on the clapper was hers, and the bell had never been silenced by a person but devoured by the city itself, and the old order shattered like glass under a hammer, and the pieces cut everyone who tried to pick them up.
# T: Mara carried the wax seal in her pocket and felt it warm against her hip, and Tobias stood at the edge of the plaza with his ledger open and his face blank, and the city rebuilt itself from raw memory, from the names that had been erased and the rhythms that had been sold and the silence that had been mistaken for peace. Sable spoke the name aloud, and the bell rang, and the sound was not a note but a word, and the word was a confession, and the confession was a door, and the door opened onto a city that had to learn to remember how to be itself.

ENC end_mix turn=0..999
# TITLE: The Sound Between
# T: Mara held the seal in one hand and the torn page in the other, and Tobias stood between them with his ledger closed and his face set, and Sable traced the ink on the clapper with a finger that would not stop trembling. The two bells rang together, and the sound that came out was neither old nor new but something in between, a sound that had no name in any language the city still spoke, and the city listened to it the way a child listens to a word they have not yet learned.
# T: The tower door opened, and the bell above rang, and the small bell below rang, and the two sounds wove together into a pattern that was not a rhythm but a shape, a shape that held the memory of the old order and the absence of the new one and the space between them where the city could build something it had not yet imagined. Mara set the seal on the counter and let it go, and Tobias closed his ledger for the last time, and Sable spoke the name into the air, and the city held its breath, and the sound held the city, and neither of them knew which of them was the bell and which was the silence.

ENC end_phase turn=0..999
# TITLE: The Silence That Was a Phase
# T: Mara opened the ledger page first and let the ink dry on her fingers, and the city's old order cracked along the lines that had been there all along, and the seal came second and the tower door opened without protest, and the name came last and the bell rang, and the sound was not a note but a word, and the word was a phase, and the phase was a turning, and the turning was a door, and the door opened onto a city that had been waiting for someone to tell it that the silence was not a loss but a breath.
# T: Tobias stood at the edge of the plaza with his ledger open and his face set, and Sable held the small bell in her hands and felt it warm against her palms, and Mara carried the wax seal in her pocket and felt it settle into her bones like a memory that had finally found its place. The city rebuilt itself from raw memory, from the names that had been erased and the rhythms that had been sold and the silence that had been mistaken for peace, and the new silence was not an absence but a listening, and the listening was a phase, and the phase was a turning, and the turning was a door, and the door opened onto a city that had to learn to remember how to be itself.
