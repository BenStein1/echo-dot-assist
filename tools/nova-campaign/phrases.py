"""Prompt pools for the wake-word recording campaign (word: "Nova").

Everything here is a script for Ben to read aloud.  A prompt is (kind, text):
  pos    the wake word, said in some context              -> label 1
  neg    something that must NOT wake the device          -> label 0
  noise  a non-speech sound to make                       -> label 0
  quiet  stay silent                                      -> label 0
Style hints are attached by campaign.py, balanced across kinds so that how a phrase is
said (whisper, far, loud...) never predicts its label.  The FINAL pools are disjoint from the
TRAIN pools and are only ever used for the held-out gate; never tune on them.
"""

POS_TRAIN = [
    "Nova", "Nova.", "Nova?", "Nova!", "Hey Nova", "Okay Nova", "Nova, are you there",
    "Nova, what time is it", "Nova, turn off the lights", "Nova, set a timer for ten minutes",
    "Nova, play some music", "Nova, what's the weather", "Nova, stop", "Nova, cancel that",
    "Nova, how long is left on the timer", "Nova, add milk to the shopping list",
    "Nova, is the front door locked", "Nova, turn on the kitchen light", "Nova, be quiet",
    "Nova, volume down", "Nova, what's on my calendar", "Nova, remind me to call mom",
    "What's the weather, Nova", "Turn on the lamp, Nova", "Can you hear me, Nova",
    "Are you listening, Nova", "Thanks, Nova", "Good morning, Nova", "Good night, Nova",
    "Please, Nova, turn it off", "So, Nova, what do you think", "Hello, Nova",
    "I said Nova", "Nova, Nova", "Nova, hello", "Nova, dot", "Hello dot, Nova",
    "Dot, dot, Nova", "Nova, no, not that one", "Nova, how are you",
    "Excuse me, Nova", "Hey, Nova, wake up", "Nova, play the news", "Nova, skip",
    "Nova, pause", "Nova, resume", "Nova, what did you say", "Nova, say that again",
    "Wait, Nova, one second", "Nova, ask me something",
]
POS_FINAL = [
    "Nova", "Nova?", "Nova, what's the time", "Nova, lights off please", "Nova, start a five minute timer",
    "Nova, play something relaxing", "Nova, is it going to rain", "Nova, stop that", "Hey Nova, hello",
    "Okay, Nova, set an alarm", "Can you help me, Nova", "Nova, turn the volume up",
    "Nova, what's next on my list", "Thanks a lot, Nova", "Nova, lock the front door",
    "Nova, how cold is it outside", "Nova, quiet please", "Yes, Nova, that one", "Nova, hello dot",
    "Nova, never mind",
]

NEG_CONFUSABLE_TRAIN = [
    # the previous false wakes
    "dot", "Dot, dot", "Hello, dot", "Dot, dot. Hello, dot.", "Hello dot, dot", "dot dot dot", "Dot.", "Dot?",
    "polka dot", "dot com", "connect the dots", "dot matrix", "dotted line", "Dotty", "dot your i's",
    "The dot is on the screen", "Dot, hello", "Hey dot", "Okay dot", "Dot, are you there",
    "Dot, what time is it", "Dot, turn off the lights", "Hello hello dot dot",
    # rhymes and near-phones of Nova / dot
    "Noah", "novel", "innovate", "innovation", "renovate", "renovation", "Nevada", "know what", "no va", "oh no",
    "no, not that", "Nolan", "Nora", "Nokia", "Novak", "Novo", "Nobel", "nobody", "noble", "no way", "snowball",
    "Jonah", "Mona", "Dora", "Cova", "hover", "sofa", "over", "mover", "lover", "Nava", "navy", "nervous", "November",
    "novice", "gnome", "know it", "know her", "own up", "oh, now", "go on", "snow", "Moda", "Dover", "Rover", "Clover",
    "Noni", "Nina", "Nana", "Nadia", "Natalia", "Noelle", "Nigel", "Nova's", "Nono", "No no no",
    "Doc", "dock", "dog", "dodge", "don't", "Dolly", "Donna", "Dan", "Dad", "Dot's", "hot dog", "not that",
    "got it", "what", "uh huh", "hmm", "oh", "no", "Noo", "Vah", "Nuh", "Dah", "Dotty dot",
    # other assistants and wake words
    "Alexa", "Echo", "Hey Google", "Hey Siri", "Computer", "Amazon", "Ziggy", "Nimbus", "Aria", "Okay Google",
    "Hey Alexa, what time is it", "Echo, play music", "Computer, lights on",
    # commands WITHOUT the wake word (must not wake)
    "turn off the lights", "what time is it", "set a timer for ten minutes", "play some music", "what's the weather",
    "stop", "cancel that", "add milk to the shopping list", "is the front door locked", "how long is left on the timer",
    "volume down", "be quiet", "never mind", "can you hear me", "are you there", "hello", "hey", "okay",
]
NEG_CONFUSABLE_FINAL = [
    "dot", "Dot, dot, hello", "Hello, dot, dot", "Hello dot", "the dot over the i", "red dot", "a tiny dot",
    "Dot, dot. Hello, dot. Dot.", "Dot, set a timer", "Dot, good morning",
    "Noah", "novelty", "innovative", "Nevada desert", "know a lot", "oh no no", "Nolan", "Nora", "snow day",
    "Dora", "over there", "no, over here", "Jonah", "dog", "Doc", "Don", "Donovan", "Dolores", "Nicole",
    "Alexa, stop", "Echo", "Hey Google, hello", "Computer", "lights off please", "what's the time",
    "play something relaxing", "is it going to rain", "hello there", "can you help me",
]

NEG_SPEECH_TRAIN = [
    "I'll be home around six if you need anything",
    "Did you see the game last night, it was unbelievable",
    "We should probably leave before the traffic gets bad",
    "I think the package should arrive on Thursday",
    "Can you pass me the remote, it's on the couch",
    "My coffee went cold again because I forgot about it",
    "The neighbor's dog was barking all morning",
    "Let's order pizza tonight, I don't feel like cooking",
    "I need to reset the router, the internet keeps dropping",
    "What did the doctor say about the test results",
    "The meeting got moved to three o'clock tomorrow",
    "Honestly I have no idea where I left my keys",
    "Remember to take the trash out before you go to bed",
    "She said she would call back after lunch",
    "It's supposed to be a hundred and ten degrees on Saturday",
    "I was reading about how those engines actually work",
    "Don't forget the charger, it's in the top drawer",
    "That movie was way longer than it needed to be",
    "We were talking about the weather",
    "Turn left at the next light and then go straight",
    "One, two, three, four, five, six, seven, eight, nine, ten",
    "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday",
    "My phone number is five five five, oh one two three",
    "Seven thirty in the morning, no wait, seven fifteen",
    "The quick brown fox jumps over the lazy dog",
    "Please remind everyone that the building closes at nine",
    "I'm going to go check on the garden",
    "How much does that cost, is it on sale",
    "Could you turn that down a little, I'm on the phone",
    "He just stood there and didn't say anything at all",
    "This is a test of the emergency broadcast system",
    "I can't believe how fast the weekend went by",
    "Where did all the plates go, I just bought a dozen",
    "What a beautiful evening, look at that sky",
    "That is the strangest thing I have heard all day",
    "I told him twice and he still didn't listen",
    "Would you rather have tacos or sandwiches",
    "I'm so tired, I could sleep for a week",
    "The new update broke everything again",
    "Hold on, let me find my glasses",
    "Is anybody home, hello, anybody there",
    "Good morning, how did you sleep",
    "Happy birthday to you, happy birthday to you",
    "Knock knock, who's there",
    "I wonder what time the store opens",
    "Somebody left the garage door open again",
    "Oh, that's a great idea, let's do that",
    "Excuse me, do you know where the bathroom is",
    "No, no, no, that's not what I meant",
    "Over there by the window, no, the other one",
    "He is a nova of talent, no wait, scratch that",  # contains the word: excluded automatically in processing
]
NEG_SPEECH_FINAL = [
    "I'll call you when I get to the airport",
    "The package is supposed to be here by Friday",
    "Honestly I think the traffic is worse than yesterday",
    "Can you grab the other chair from the kitchen",
    "My coffee machine stopped working this morning",
    "Did anyone see where the dog went",
    "The new neighbors seem really friendly",
    "We should plan something for the long weekend",
    "I need to renew my passport before the trip",
    "What time does the game start tonight",
    "She left the keys on the counter again",
    "Nine, eight, seven, six, five, four, three, two, one",
    "January, February, March, April, May, June",
    "The rain in Spain stays mainly in the plain",
    "Don't worry about it, I'll take care of it",
    "That is not what I ordered, I asked for the small one",
    "Hello, hello, is this thing on",
    "Wow, look how late it is already",
    "I love this song, turn it up a little",
    "Maybe we could watch something funny tonight",
]

NOISE = [
    "clap twice", "clap once, sharply", "cough", "clear your throat", "laugh", "sneeze (or fake one)", "sigh heavily",
    "jingle your keys", "tap the table", "tap the top of the Dot", "knock on the wall", "whistle a short tune",
    "hum a few notes", "sing a line of any song", "say 'psst'", "say 'shhh'", "say 'um... uh... hmm'", "yawn",
    "sniff", "snap your fingers", "drop something light on the table", "close a drawer or a door", "stir a spoon in a cup",
    "rustle some paper", "breathe loudly close to the Dot", "say 'ha ha ha'", "clink two glasses",
    "slide a chair", "walk past the Dot", "type on a keyboard",
]
NOISE_FINAL = [
    "clap three times", "cough twice", "laugh", "clear your throat", "jingle keys", "tap the table",
    "whistle", "hum", "sigh", "walk past the Dot", "close a door", "rustle paper",
]

READ_PARAGRAPHS = [
    "The old house at the end of the street had been empty for years. Every autumn the leaves piled up on the porch, "
    "and every spring somebody promised to clean them away. Nobody ever did. Children said the windows watched them "
    "as they walked to school, but the windows only reflected the clouds. One morning a truck arrived, and two men "
    "carried boxes through the front door. By evening a lamp glowed in the upstairs room, and the street felt a little "
    "less quiet than it had been before.",
    "To make good bread you need flour, water, salt, and patience. Mix everything until it forms a rough dough, "
    "then let it rest. After an hour fold it over itself a few times, cover it with a cloth, and wait again. "
    "The dough will slowly double in size. Shape it gently, because you don't want to knock out all the air. "
    "Bake it in a hot oven until the crust turns deep brown and sounds hollow when you tap it. "
    "Then, hardest of all, let it cool before you cut into it.",
    "Last summer we drove across the desert to see the mountains. The road was straight for miles, and the radio "
    "kept losing the signal. We stopped at a gas station that sold ice cream, maps, and rubber snakes. "
    "The man at the counter told us about a trail with a waterfall, and we decided to try it. "
    "It took longer than we expected, and we were hot and tired, but the water was cold and clear, "
    "and for a while nobody said a word. That was the best part of the whole trip.",
]
