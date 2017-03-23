# game2vec - games embeddings 

TensorFlow implementation of word2vec applied on https://www.kaggle.com/tamber/steam-video-games dataset, using both CBOW and Skip-gram.

Context for each game is extracted from the other games that the user owns. For example if a user has three games: Dota 2, CS: GO, and Rocket League, this (input -> label) pairs can be generated:

 * CBOW: ((Dota 2, CS: GO) -> Rocket League), ((Dota 2, Rocket League) -> CS: GO), ((CS: GO, Rocket League) -> Dota 2)
 * Skip-gram: (Rocket League -> (Dota 2, CS: GO)), (CS: GO -> (Dota 2, Rocket League)), (Dota 2 -> (CS: GO, Rocket League))


For more reference, please have a look at this papers:
 
 * [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
 * [word2vec Parameter Learning Explained](http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf)
 * [Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method](http://arxiv.org/pdf/1402.3722v1.pdf)
 
There are three training scripts:

 * **train_cbow.py** - training using CBOW, using both purchase and play actions into account as user context.
 * **train_cbow_weighted.py** - same as above, but, only play actions are taken into consideration, and the label is selected based on time played (more time played the game - higher the probability of being selected).
 * **train_skipgram.py** - training using Skip-gram, using both purchase and play actions into account as user context.
 
Each script outputs an image with the game embeddings visualised using t-SNE, and a .npy file containing embeddings, dictionary, and reverse dictionary:
```
{
    'embeddings': final_embeddings,
    'idx_to_game': idx_to_game,
    'game_to_idx': game_to_idx
}
```

There are three notebooks that load the embeddings + some examples:

 * **player_cbow.py** - loads embeddings_cbow.npy + example usage. 
 * **player_cbow_weighted.py** - loads embeddings_cbow_weighted.npy + example usage.
 * **player_skipgram.py** - loads embeddings_skipgram.npy + example usage.
 
Visualisations are found in .png files.

Example training output:
```
There are 3600 unique games in the data set.
[(24, 9.9), (109, 2.9), (9, 1.3), (273, 0.7), (237, 6.2), (234, 58.0), (307, 10.8), (120, 10.6), (77, 0.5)]
["Garry's Mod", 'Half-Life 2 Episode One', 'Team Fortress 2', 'Counter-Strike', 'Day of Defeat Source', 'Counter-Strike Source', 'Audiosurf', 'Portal', 'Half-Life 2']
[(238, 316.0)]
['Empire Total War']
[(241, 2.2), (234, 304.0), (9, 0.1), (580, 4.7), (237, 0.9)]
['Alien Swarm', 'Counter-Strike Source', 'Team Fortress 2', 'Half-Life 2 Deathmatch', 'Day of Defeat Source']
[(1983, 0.4)]
['Snuggle Truck']
240 -> [1457]
Unturned -> Smashmuck Champions
9 -> [1457]
Team Fortress 2 -> Smashmuck Champions
240 -> [1457]
Unturned -> Smashmuck Champions
819 -> [1457]
Trove -> Smashmuck Champions
9 -> [1457]
Team Fortress 2 -> Smashmuck Champions
240 -> [1457]
Unturned -> Smashmuck Champions
9 -> [1457]
Team Fortress 2 -> Smashmuck Champions
819 -> [1457]
Trove -> Smashmuck Champions
[ 0  0  1  1  2  2  3  3  4  4  5  5  6  6  7  7  8  8  9  9 10 10 11 11 12
 12 13 13 14 14 15 15 16 16 17 17 18 18 19 19 20 20 21 21 22 22 23 23 24 24
 25 25 26 26 27 27 28 28 29 29 30 30 31 31 32 32 33 33 34 34 35 35 36 36 37
 37 38 38 39 39 40 40 41 41 42 42 43 43 44 44 45 45 46 46 47 47 48 48 49 49
 50 50 51 51 52 52 53 53 54 54 55 55 56 56 57 57 58 58 59 59 60 60 61 61 62
 62 63 63]
weights: [3600, 128]
biases: [3600]
final_embed: [None, 128]
Initialised
Average loss at step 0 : 104.436584473
Nearest to Company of Heroes: Gods Will Be Watching, Cloud Chamber, Oil Rush, Stealth Inc 2, Mordheim City of the Damned, FINAL FANTASY XIII-2, Amazing World, Perpetuum,
Nearest to Deus Ex Human Revolution: Duke Nukem Forever, Tom Clancy's H.A.W.X. 2, Full Mojo Rampage, Freight Tycoon Inc., GRID Autosport, 100% Orange Juice, Gunscape, Deus Ex Game of the Year Edition,
Nearest to Mirror's Edge: Critter Crunch, Lost Planet Extreme Condition - Colonies Edition, Tom Clancy's Splinter Cell Chaos Theory, Operation Flashpoint Red River, Year Walk, Ys I, Petz Horsez 2, Third Eye Crime,
Nearest to Counter-Strike Global Offensive: Majesty 2, Nancy Drew Danger on Deception Island , Amazing Adventures Around the World, X2 The Threat, Pirates, Vikings, & Knights II, Stargate Resistance, Heroes Over Europe, Duck Dynasty,
Nearest to Far Cry 3: Ubersoldier II, BookWorm Deluxe, Bridge Constructor Medieval, Nimble Writer, Fieldrunners 2, Carpe Diem, Dark Void, Sacred Gold,
Nearest to Team Fortress 2: Stronghold 2, Nexus The Jupiter Incident, Larva Mortus, Cossacks II Napoleonic Wars, Edge of Space, Don Bradman Cricket 14, Samantha Swift and the Golden Touch, Worms Crazy Golf,
Nearest to Fallout 4: Kinetic Void, Viscera Cleanup Detail Shadow Warrior, Alice Madness Returns, Gratuitous Space Battles, Desktop Dungeons, Peggle Extreme, Star Wars - Battlefront II, The Misadventures of P.B. Winterbottom,
Nearest to Batman Arkham Asylum GOTY Edition: WRC Powerslide, Bridge Constructor, World of Guns Gun Disassembly, the static speaks my name, The Curious Expedition, Fate of the World, Dead Mountaineer's Hotel, Undead Overlord,
Nearest to Half-Life 2: Railroad Pioneer, The Book of Unwritten Tales, Torchlight II, Endless Sky, One Way Heroics, Cabela's Big Game Hunter Pro Hunts, Resident Evil 5 / Biohazard 5, Medal of Honor(TM) Single Player,
Nearest to Orcs Must Die!: Gauntlet , Altitude, Polarity, Lume, Captain Forever Remix, TRI Of Friendship and Madness, Nightmares from the Deep 2 The Siren`s Call, Anno 2205,
Nearest to Space Pirates and Zombies: Doctor Who The Eternity Clock, Montague's Mount, Back to the Future Ep 2 - Get Tannen!, Twin Sector, Medieval II Total War Kingdoms, 100% Orange Juice, Alien Breed 2 Assault, Eternal Silence,
Nearest to FINAL FANTASY VII: Industry Giant 2, Anomaly 2, Escape Goat, Pirates of Black Cove, Nuclear Dawn, Insanely Twisted Shadow Planet, Desperados - Wanted Dead or Alive, Movie Studio 13 Platinum - Steam Powered,
Nearest to Saints Row IV: Arms Dealer, CroNix, Echelon, Sid Meier's Ace Patrol Pacific Skies, GunWorld, Wildlife Park 2, Intrusion 2, Avadon The Black Fortress,
Nearest to Fallout 3 - Game of the Year Edition: Project AURA, Guild of Dungeoneering, Child of Light, Defy Gravity, Minimon, SC2VN - The eSports Visual Novel, Nancy Drew Ghost of Thornton Hall, Thief Deadly Shadows,
Nearest to Dragon Age Origins - Ultimate Edition: Castlevania Lords of Shadow  Mirror of Fate HD, Vertiginous Golf, Sonic Adventure 2 , Patrician IV Steam Special Edition, bit Dungeon II, Swarm Arena, Zuma's Revenge! - Adventure, Fall of the New Age Premium Edition,
Nearest to PROTOTYPE 2: Football Tactics, DARK SOULS II Scholar of the First Sin, The Night of the Rabbit, Mutant Mudds Deluxe, Dungeon Defenders Eternity, Sam & Max 301 The Penal Zone, Crusader Kings II, Duke Nukem Forever ,
Average loss at step 2000 : 34.399090982
Average loss at step 4000 : 17.6512645052
Average loss at step 6000 : 14.1487481684
Average loss at step 8000 : 12.3442589823
Average loss at step 10000 : 11.3791581116
Nearest to Company of Heroes: Star Wars Dark Forces, Weather Lord The Successor's Path, Cloud Chamber, Freedom Planet, VVVVVV, Secret Of Magia, Dungeon Defenders, Habitat,
Nearest to Deus Ex Human Revolution: Homefront, Infinifactory, Fable - The Lost Chapters, War Thunder, 15 Days, Written in the Sky, Edge of Space, Faerie Solitaire,
Nearest to Mirror's Edge: Fingered, Canyon Capers, Floating Point, Resident Evil 5 / Biohazard 5, Slender The Arrival, Viscera Cleanup Detail Shadow Warrior, Always Sometimes Monsters, Grand Theft Auto V,
Nearest to Counter-Strike Global Offensive: Team Fortress 2, Fallout 4, Bus-Simulator 2012, SpeedRunners, Lunar Flight, Nosgoth, Left 4 Dead 2, Garry's Mod,
Nearest to Far Cry 3: Call of Duty Ghosts - Multiplayer, Shift 2 Unleashed, The Repopulation, Steel Storm Burning Retribution, RollerCoaster Tycoon 3 Platinum!, Third Eye Crime, Intergalactic Bubbles, Far Cry 4,
Nearest to Team Fortress 2: Counter-Strike Global Offensive, Unturned, Contrast, Fable - The Lost Chapters, Abducted, Driftmoon, Cultures - Northland, Football Manager 2015,
Nearest to Fallout 4: Counter-Strike Global Offensive, Devils Share, Unturned, Bob Came in Pieces, Galactic Civilizations I Ultimate Edition, The Clans - Saga of the Twins, Indigo Prophecy, Deadlings - Rotten Edition,
Nearest to Batman Arkham Asylum GOTY Edition: Mars Colony Challenger, Blockstorm, Rome Total War - Alexander, Deathmatch Classic, Fritz Chess 14, Cabela's African Adventures, LEGO Batman 2, Hatred,
Nearest to Half-Life 2: Intrusion 2, ORION Prelude, TDP5 Arena 3D, Runaway A Road Adventure, Far Cry, Avernum 4, Bloodbath Kavkaz, Mortal Kombat X,
Nearest to Orcs Must Die!: GUILTY GEAR XX ACCENT CORE PLUS R, Call of Duty Modern Warfare 2 - Multiplayer, Obscure 2, DOOM 3 BFG Edition, Max The Curse of Brotherhood, Monday Night Combat, Hyper Fighters, Secrets of Rtikon,
Nearest to Space Pirates and Zombies: DiRT 3 Complete Edition, Darkest Dungeon, Millie, Snapshot, Deathtrap, It came from space, and ate our brains, Else Heart.Break(), Quake Mission Pack 1 Scourge of Armagon,
Nearest to FINAL FANTASY VII: South Park The Stick of Truth, The Flame in the Flood, Eufloria HD, Ice Age Continental Drift Arctic Games, Bus-Simulator 2012, Ghostbusters Sanctum of Slime, Singularity, Amazing Adventures Around the World,
Nearest to Saints Row IV: Verdun, Cities Skylines, RollerCoaster Tycoon 3 Platinum!, The Expendabros, Hegemony Philip of Macedon, Blitzkrieg Anthology, TRI Of Friendship and Madness, The Sims(TM) 3,
Nearest to Fallout 3 - Game of the Year Edition: Prince of Persia The Two Thrones, Dust An Elysian Tail, Fieldrunners, Prison Tycoon 3 Lockdown, Incredipede, Nihilumbra, State of Decay Year-One, War of Beach,
Nearest to Dragon Age Origins - Ultimate Edition: Antichamber, NBA 2K15, Car Mechanic Simulator 2015, Ghostbusters Sanctum of Slime, Wickland, New kind of adventure, Homeworld Remastered Collection, Armello,
Nearest to PROTOTYPE 2: Offworld Trading Company, FIFA Manager 09, East India Company Gold, Delta Force Black Hawk Down, Buccaneer The Pursuit of Infamy, Arma 2 Operation Arrowhead Beta (Obsolete), Fallout 3, Agarest Zero,
Average loss at step 12000 : 9.93935323229
Average loss at step 14000 : 9.20145629555
Average loss at step 16000 : 8.79739663076
Average loss at step 18000 : 8.00816985276
Average loss at step 20000 : 7.85464020553
Nearest to Company of Heroes: Steel Storm Burning Retribution, Iron Warriors T-72 Tank Command, Sacred Gold, Crusader Kings II, Orcs Must Die! 2, Jack Lumber, Build-A-Lot 2, Serena,
Nearest to Deus Ex Human Revolution: Half-Life 2 Lost Coast, Jolly Rover, Velvet Assassin, Alien Breed Impact, Amazing Princess Sarah, SpeedRunners, Dragon Nest Europe, Sacred 2 Gold,
Nearest to Mirror's Edge: Giana Sisters Twisted Dreams, Blockscape, Piercing Blow, Razor2 Hidden Skies, TerraTech, CRYPTARK, March of War, Fingered,
Nearest to Counter-Strike Global Offensive: Dota 2, Left 4 Dead 2, Garry's Mod, War Thunder, No More Room in Hell, Counter-Strike Condition Zero, Dying Light, Arma 2,
Nearest to Far Cry 3: Call of Duty Modern Warfare 2 - Multiplayer, Codename Panzers - Cold War, Aerena, EverQuest Secrets of Faydwer, Talisman Digital Edition, The Black Watchmen, Build-A-Lot 2, Painkiller Recurring Evil,
Nearest to Team Fortress 2: Cubemen 2, War Thunder, Alien Swarm, Counter-Strike, Dota 2, Metaverse Construction Kit, Unturned, King Arthur Collection,
Nearest to Fallout 4: Crusader Kings II, Servo, Blockscape, Who's That Flying?!, Dungeons & Dragons Online, SpellForce 2 - Faith in Destiny, Toki Tori 2+, Death Skid Marks,
Nearest to Batman Arkham Asylum GOTY Edition: 10 Second Ninja, Commandos 2 Men of Courage, Elite Dangerous Horizons, Huntsman - The Orphanage Halloween Edition, BioShock Infinite, Beatbuddy Tale of the Guardians, Majesty Gold Edition, Burnout Paradise The Ultimate Box,
Nearest to Half-Life 2: Half-Life 2 Lost Coast, Portal, Half-Life 2 Episode Two, Counter-Strike Source, Audiosurf, Distance, Half-Life, Portal 2,
Nearest to Orcs Must Die!: Towns, Lakeview Cabin Collection, Endless Sky, Dungeons & Dragons Daggerdale, Mount & Blade Warband, CRYPTARK, Alter World, Bus-Simulator 2012,
Nearest to Space Pirates and Zombies: Crystals of Time, Mark of the Ninja, Banished, Wolfenstein The New Order, DiRT 3 Complete Edition, It came from space, and ate our brains, Snapshot, March of War,
Nearest to FINAL FANTASY VII: Crazy Machines 2, Solstice Arena, The Flame in the Flood, how do you Do It?, Dawn of Discovery - Venice, Beatbuddy Tale of the Guardians, Abyss The Wraiths of Eden, Ghostbusters Sanctum of Slime,
Nearest to Saints Row IV: Armello, Rise of Incarnates, Section 8, Assassin's Creed Revelations, The Flame in the Flood, LEGO The Lord of the Rings, Jagged Alliance Flashback, Divinity Dragon Commander,
Nearest to Fallout 3 - Game of the Year Edition: Sakura Spirit, State of Decay Year-One, Tom Clancy's Ghost Recon Phantoms - NA, Magic The Gathering  Duels of the Planeswalkers 2012, Dead Pixels, Freddi Fish and Luther's Maze Madness, FINAL FANTASY XI, Call of Duty Modern Warfare 3 - Multiplayer,
Nearest to Dragon Age Origins - Ultimate Edition: Peggle Extreme, Super Meat Boy, Divinity Original Sin, Grey Goo, Might & Magic Heroes VI, Zombie Panic Source, SCHAR Blue Shield Alliance, Call of Duty Black Ops II - Multiplayer,
Nearest to PROTOTYPE 2: FIFA Manager 09, Call of Duty Ghosts, Delta Force Black Hawk Down, Maia, Buccaneer The Pursuit of Infamy, EverQuest Secrets of Faydwer, SlamIt Pinball Big Score, Galactic Civilizations I Ultimate Edition,
Average loss at step 22000 : 7.51871938872
Average loss at step 24000 : 7.26883857368
Average loss at step 26000 : 7.10217741412
Average loss at step 28000 : 6.70527764418
Average loss at step 30000 : 6.50435507616
Nearest to Company of Heroes: Tropico 5, Habitat, Face Noir, Fading Hearts, Dark Souls Prepare to Die Edition, Warhammer 40,000 Dawn of War  Winter Assault, Divekick, Dungeons The Eye of Draconus,
Nearest to Deus Ex Human Revolution: Jolly Rover, Fallout 3 - Game of the Year Edition, David., Zafehouse Diaries, Carmageddon 2 Carpocalypse Now, Call of Duty Modern Warfare 3, Medieval Engineers, Time Mysteries 2 The Ancient Spectres,
Nearest to Mirror's Edge: Borderlands 2, Far Cry 3 Blood Dragon, NARUTO SHIPPUDEN Ultimate Ninja STORM Revolution, Left 4 Dead 2, Borderlands, The Hat Man Shadow Ward, LEGO MARVEL Super Heroes, Might & Magic Heroes Online,
Nearest to Counter-Strike Global Offensive: Team Fortress 2, Dead Island Epidemic, Left 4 Dead 2, PAYDAY 2, Unturned, Dota 2, Worms Clan Wars, Counter-Strike Condition Zero,
Nearest to Far Cry 3: Constant C, Serena, Left 4 Dead, PAYDAY 2, Hitman Absolution, DiggerOnline, Europa Universalis IV, Counter-Strike Condition Zero Deleted Scenes,
Nearest to Team Fortress 2: Counter-Strike Global Offensive, Unturned, Counter-Strike Source, Left 4 Dead 2, Dota 2, No More Room in Hell, Counter-Strike Nexon Zombies, APB Reloaded,
Nearest to Fallout 4: The Elder Scrolls V Skyrim, Dark Fall 1 The Journal, Blue Rose, Bad Rats, Larva Mortus, Metro 2033, Fieldrunners 2, The Forest,
Nearest to Batman Arkham Asylum GOTY Edition: Kingdoms of Amalur Reckoning, The Cat Lady, Commandos 2 Men of Courage, Huntsman - The Orphanage Halloween Edition, Magic The Gathering - Duels of the Planeswalkers 2013, Cricket Revolution, Spirits, Medieval II Total War,
Nearest to Half-Life 2: Half-Life 2 Lost Coast, Fallout 3, Half-Life 2 Deathmatch, Unturned, Trine, Paint the Town Red, Counter-Strike Source, Sniper Elite,
Nearest to Orcs Must Die!: Ether Vapor Remaster, Railroad Tycoon 3, Football Manager 2011, Cabela's Big Game Hunter Pro Hunts, Oh...Sir!, Grim Legends 2 Song of the Dark Swan, The Elder Scrolls III Morrowind, E.Y.E Divine Cybermancy,
Nearest to Space Pirates and Zombies: Crystals of Time, Euro Truck Simulator 2, Airport Madness World Edition, Atlantica Online, DUNGEONS - Steam Special Edition, METAL SLUG 3, Quake Mission Pack 1 Scourge of Armagon, It came from space, and ate our brains,
Nearest to FINAL FANTASY VII: Crazy Machines 2, The Settlers 7 Paths to a Kingdom - Gold Edition, Pretty Girls Mahjong Solitaire, Ninja Reflex Steamworks Edition, DARK SOULS II Scholar of the First Sin, The Elder Scrolls V Skyrim, CreaVures, Tomb Raider II,
Nearest to Saints Row IV: Minimon, Sol Survivor, Dungeons The Eye of Draconus, Section 8, Hamlet or the last game without MMORPG features, shaders and product placement, Starpoint Gemini 2, Grand Theft Auto V, Tom Clancy's H.A.W.X. 2,
Nearest to Fallout 3 - Game of the Year Edition: Alien Breed Impact, World in Conflict Soviet Assault, Deus Ex Human Revolution, 10 Second Ninja, Crysis, Call of Duty Black Ops - Multiplayer, Penny Arcade's On the Rain-Slick Precipice of Darkness 4, Light,
Nearest to Dragon Age Origins - Ultimate Edition: Ironclad Tactics, Kairo, Fable - The Lost Chapters, Silverfall, NBA 2K15, Sniper Elite V2, Clash of Puppets, Metro 2033 Redux,
Nearest to PROTOTYPE 2: FIFA Manager 09, Maia, Kerbal Space Program, SlamIt Pinball Big Score, EverQuest Secrets of Faydwer, FINAL FANTASY XI, Shadow Harvest Phantom Ops, Arma 2 Operation Arrowhead,
Average loss at step 32000 : 6.29452444088
Average loss at step 34000 : 6.19835431169
Average loss at step 36000 : 6.09576434309
Average loss at step 38000 : 5.71168697058
Average loss at step 40000 : 6.02205963767
Nearest to Company of Heroes: TrackMania Stadium, Homefront, Face Noir, Empire Total War, Half-Life Source, Jade Empire Special Edition, Cabela's Dangerous Hunts 2013, Legend of Mysteria,
Nearest to Deus Ex Human Revolution: Sniper Elite 3, Sleeping Dogs, Half-Life 2 Episode Two, H1Z1, Dead Space, Fritz Chess 14, Dead Island, Alan Wake,
Nearest to Mirror's Edge: Call of Duty Black Ops, Dragon Nest Europe, Grim Legends The Forsaken Bride, DOOM II Hell on Earth, Real Warfare, Counter-Strike Condition Zero, Doc Clock The Toasted Sandwich of Time, King's Bounty The Legend,
Nearest to Counter-Strike Global Offensive: Team Fortress 2, Unturned, Aftermath, Dota 2, War Thunder, Warframe, Rocket League, PAYDAY 2,
Nearest to Far Cry 3: Call of Duty Modern Warfare 3 - Multiplayer, Akane the Kunoichi, Metro Last Light, Dead State, Space Pirates and Zombies, Space Run, Mark of the Ninja, Chime,
Nearest to Team Fortress 2: Counter-Strike Global Offensive, Dota 2, Unturned, Garry's Mod, War Thunder, Counter-Strike Source, Robocraft, Warframe,
Nearest to Fallout 4: Saw, The Forest, S.T.A.L.K.E.R. Call of Pripyat, Cargo Commander, The Witcher Enhanced Edition, ARK Survival Evolved, Castle Crashers, The Elder Scrolls V Skyrim,
Nearest to Batman Arkham Asylum GOTY Edition: aerofly RC 7, Beatbuddy Tale of the Guardians, Ironcast, Cloudbuilt, Majesty 2, Final Slam 2, No Time to Explain, Sam & Max 205 What's New Beelzebub?,
Nearest to Half-Life 2: Day of Defeat Source, Half-Life 2 Deathmatch, Half-Life 2 Lost Coast, Fallout 3, Team Fortress 2, Alien Swarm, Garry's Mod, Dota 2,
Nearest to Orcs Must Die!: DOOM 3 BFG Edition, Lunar Flight, Wargame European Escalation, Dirty Bomb, Rake, Orcs Must Die! 2, Chivalry Medieval Warfare, The Binding of Isaac,
Nearest to Space Pirates and Zombies: Crystals of Time, DUNGEONS - Steam Special Edition, Far Cry 3, Darwinia, Airport Madness World Edition, Dead Effect, Abducted, Painkiller Hell & Damnation,
Nearest to FINAL FANTASY VII: Darksiders II, Crazy Machines 2, Ragnarok Online 2, Spoiler Alert, BattleSpace, Contrast, Penny Arcade's On the Rain-Slick Precipice of Darkness 3, Victim of Xen,
Nearest to Saints Row IV: Plants vs. Zombies Game of the Year, Firefly Online Cortex, OMSI 2, Portal 2, Counter-Strike Source, The Settlers 7 Paths to a Kingdom, Brick-Force, Hyperdimension Neptunia Re;Birth1,
Nearest to Fallout 3 - Game of the Year Edition: Just Cause 2, Epic Arena, King Arthur's Gold, Serious Sam 3 BFE, Natural Selection 2, War of the Roses, Sid Meier's Starships, BioShock,
Nearest to Dragon Age Origins - Ultimate Edition: The Guild II - Pirates of the European Seas, Kairo, Darkest Dungeon, Empire Total War, Fable - The Lost Chapters, Big Brain Wolf, Clash of Puppets, Star Wars - Battlefront II,
Nearest to PROTOTYPE 2: FIFA Manager 09, X-COM Interceptor, Kerbal Space Program, One Finger Death Punch, Maia, Dead Pixels, EverQuest Secrets of Faydwer, Lost Planet 2,
Average loss at step 42000 : 5.77674743121
Average loss at step 44000 : 5.61901905833
Average loss at step 46000 : 5.40736982761
Average loss at step 48000 : 5.51045300324
Average loss at step 50000 : 5.35962552965
Nearest to Company of Heroes: TrackMania Stadium, Yet Another Zombie Defense, Tropico 5, Jade Empire Special Edition, Legend of Mysteria, Medieval II Total War, Half-Life Source, Half-Life 2 Lost Coast,
Nearest to Deus Ex Human Revolution: Borderlands 2, Flesh Eaters, DarkStar One, Just Cause 2, The Elder Scrolls V Skyrim, Fallout 4, Counter-Strike Global Offensive, Men of War,
Nearest to Mirror's Edge: Fable - The Lost Chapters, Far Cry 3 Blood Dragon, DuckTales Remastered, Infestation Survivor Stories, Dead Space 2, Skyward Collapse, Black Fire, Memories of a Vagabond,
Nearest to Counter-Strike Global Offensive: Dota 2, Team Fortress 2, Rocket League, Dirty Bomb, Grand Theft Auto V, Unturned, PAYDAY 2, PAYDAY The Heist,
Nearest to Far Cry 3: Sid Meier's Civilization Beyond Earth, NightSky, F1 2013, X-COM UFO Defense, METAL SLUG 3, Fallout New Vegas, Grand Theft Auto San Andreas, Divinity Dragon Commander Beta,
Nearest to Team Fortress 2: Dota 2, Counter-Strike Global Offensive, Counter-Strike Source, Garry's Mod, Unturned, PAYDAY 2, Killing Floor, The Elder Scrolls V Skyrim,
Nearest to Fallout 4: Half-Life 2, Morphopolis, Dark Fall 1 The Journal, Metro 2033, The Cat Lady, The Elder Scrolls V Skyrim, Call of Duty Modern Warfare 3, Deus Ex Human Revolution,
Nearest to Batman Arkham Asylum GOTY Edition: Left 4 Dead 2, Serious Sam 2, Tom Clancy's Ghost Recon Advanced Warfighter, The Chaos Engine, BioShock, The Elder Scrolls IV Oblivion , Keen Dreams, Fallout New Vegas,
Nearest to Half-Life 2: Half-Life 2 Episode One, Counter-Strike Source, Half-Life 2 Episode Two, Portal 2, Half-Life 2 Lost Coast, Day of Defeat Source, Fallout 4, AdVenture Capitalist,
Nearest to Orcs Must Die!: Torchlight II, Lunar Flight, Counter-Strike Condition Zero, Strife, DOOM 3 BFG Edition, Doodle God, Deus Ex Human Revolution, Bastion,
Nearest to Space Pirates and Zombies: Crystals of Time, Jamestown, Fallout 3 - Game of the Year Edition, Mark of the Ninja, DUNGEONS - Steam Special Edition, METAL SLUG 3, Darwinia, Cloud Chamber,
Nearest to FINAL FANTASY VII: Ninja Reflex Steamworks Edition, Motorbike, Age of Empires III Complete Collection, Left 4 Dead 2, Asteria, Original War, Call of Duty Black Ops II - Multiplayer, Sunless Sea,
Nearest to Saints Row IV: Child of Light, Fingered, Grand Theft Auto V, H1Z1, Borderlands, Tribes Ascend, Counter-Strike Source, Garry's Mod,
Nearest to Fallout 3 - Game of the Year Edition: Counter-Strike Source, Sid Meier's Civilization V, Black Mesa, Orbital Gear, Survivalist, Fallout New Vegas, Into the Dark, Space Pirates and Zombies,
Nearest to Dragon Age Origins - Ultimate Edition: Robocraft, Skullgirls, Clash of Puppets, The Tiny Bang Story, Counter-Strike, Killing Floor, Dungeon Defenders, AaaaaAAaaaAAAaaAAAAaAAAAA!!! for the Awesome,
Nearest to PROTOTYPE 2: RAGE, F1 2013, FIFA Manager 09, Baldur's Gate II Enhanced Edition, RPG Maker VX Ace, DayZ, Sniper Elite V2, ACE COMBAT ASSAULT HORIZON Enhanced Edition,
Average loss at step 52000 : 5.23387034962
Average loss at step 54000 : 5.15490737957
Average loss at step 56000 : 5.02347424981
Average loss at step 58000 : 5.04178190088
Average loss at step 60000 : 5.10526000699
Nearest to Company of Heroes: Legend of Mysteria, Total War ROME II - Emperor Edition, Into the Dark, iBomber Defense Pacific, Yet Another Zombie Defense, Special Forces Team X, Assassin's Creed II, Cossacks Back to War,
Nearest to Deus Ex Human Revolution: Borderlands 2, Sniper Ghost Warrior 2, Fallout 4, Call of Duty Advanced Warfare, Call of Duty Modern Warfare 3 - Multiplayer, Sniper Elite 3, INK, Empire Total War,
Nearest to Mirror's Edge: Wasteland 1 - The Original Classic, ENSLAVED Odyssey to the West Premium Edition, Call of Duty Black Ops, Rome Total War, Mighty Gunvolt, FEZ, LYNE, 10 Second Ninja,
Nearest to Counter-Strike Global Offensive: Ace of Spades, Dota 2, Team Fortress 2, PAYDAY 2, Garry's Mod, Unturned, War Thunder, Grand Theft Auto V,
Nearest to Far Cry 3: The Talos Principle, Left 4 Dead, The Club, METAL SLUG 3, ONE PIECE PIRATE WARRIORS 3, Half-Life 2 Episode One, Wasteland Angel, Sid Meier's Civilization V,
Nearest to Team Fortress 2: Unturned, Counter-Strike Global Offensive, Sid Meier's Civilization V, Counter-Strike Source, Alien Swarm, Ace of Spades, War Thunder, Dirty Bomb,
Nearest to Fallout 4: DayZ, Tomb Raider, Call of Duty Advanced Warfare - Multiplayer, Deus Ex Human Revolution, Arma 2 Operation Arrowhead, Age of Empires II HD Edition, The Cat Lady, Insaniquarium! Deluxe,
Nearest to Batman Arkham Asylum GOTY Edition: Gurumin A Monstrous Adventure, RAGE, Ghost Encounters Deadwood - Collector's Edition, Mortal Kombat Kollection, Angry Video Game Nerd Adventures, Incredipede, Chessmaster, Commandos 2 Men of Courage,
Nearest to Half-Life 2: Half-Life 2 Episode One, Day of Defeat Source, Half-Life 2 Lost Coast, Half-Life 2 Episode Two, Sid Meier's Civilization V, Fallout, Counter-Strike Source, Legend of Dungeon,
Nearest to Orcs Must Die!: Borderlands 2, Waveform, Football Manager 2011, Into the Dark, Left 4 Dead 2, Spiral Knights, Viridi, Space Colony,
Nearest to Space Pirates and Zombies: METAL SLUG 3, Darwinia, One Way Heroics, Close Your Eyes, Mark of the Ninja, Fallout 3 - Game of the Year Edition, Crystals of Time, Tomb Raider Anniversary,
Nearest to FINAL FANTASY VII: The Secret of Monkey Island Special Edition, Dragon Quest Heroes, Hack, Slash, Loot, Age of Empires III Complete Collection, Skulls of the Shogun, Ninja Reflex Steamworks Edition, Blacklight Tango Down, Heavy Bullets,
Nearest to Saints Row IV: Styx Master of Shadows, Grand Theft Auto V, Fading Hearts, Nancy Drew Danger on Deception Island , Jagged Alliance Crossfire, Warhammer 40,000 Dawn of War  Winter Assault, The Expendabros, Max The Curse of Brotherhood,
Nearest to Fallout 3 - Game of the Year Edition: Counter-Strike Source, Borderlands 2, The Elder Scrolls V Skyrim, Fallout New Vegas, Tomb Raider, SimCity 4 Deluxe, Fable - The Lost Chapters, Rust,
Nearest to Dragon Age Origins - Ultimate Edition: The Tiny Bang Story, Magicka, Far Cry 3 Blood Dragon, BioShock, Half-Life 2 Deathmatch, Joe Danger, SUPER DISTRO, Might & Magic Heroes VI,
Nearest to PROTOTYPE 2: FIFA Manager 09, Watchmen The End Is Nigh Part 2, F1 2013, RAGE, FINAL FANTASY XI, Controller Companion, Max Payne 3, RPG Maker VX Ace,
Average loss at step 62000 : 5.04800261237
Average loss at step 64000 : 4.82001546143
Average loss at step 66000 : 4.79032419768
Average loss at step 68000 : 4.79491334222
Average loss at step 70000 : 4.60082254729
Nearest to Company of Heroes: Into the Dark, Project CARS, Company of Heroes (New Steam Version), Keep Talking and Nobody Explodes, Besiege, Resident Evil 6 / Biohazard 6, iBomber Defense Pacific, Civil War II,
Nearest to Deus Ex Human Revolution: Call of Duty Black Ops, Amazing Princess Sarah, Half-Life 2 Episode One, Alan Wake, D.U.S.T., Dishonored, Fallout 3 - Game of the Year Edition, Risen 2 - Dark Waters,
Nearest to Mirror's Edge: Infestation Survivor Stories, Thief Town, Titan Quest Immortal Throne, Rogue Shooter The FPS Roguelike, Silent Hill Homecoming, Defiance, Warside, Left 4 Dead 2,
Nearest to Counter-Strike Global Offensive: Dota 2, Left 4 Dead 2, Ace of Spades, Counter-Strike Source, Chivalry Medieval Warfare, Rocket League, Warframe, Half-Life 2 Deathmatch,
Nearest to Far Cry 3: Goodbye Deponia, METAL GEAR SOLID V GROUND ZEROES, Tomb Raider, The Club, Beatbuddy Tale of the Guardians, MLB 2K11, Take Command Second Manassas, Puzzle Agent 2,
Nearest to Team Fortress 2: Left 4 Dead 2, Unturned, Dota 2, Counter-Strike Source, Robocraft, Left 4 Dead, Garry's Mod, Injustice Gods Among Us Ultimate Edition,
Nearest to Fallout 4: Warhammer 40,000 Dawn of War II  Retribution, Arma 2 DayZ Mod, Sid Meier's Civilization V, Blue Rose, The Cat Lady, Metro Last Light, Divekick, Darkwind War on Wheels,
Nearest to Batman Arkham Asylum GOTY Edition: Flora's Fruit Farm, Team Fortress 2, Batman Arkham City GOTY, Ryse Son of Rome, Gurumin A Monstrous Adventure, Tom Clancy's Ghost Recon Advanced Warfighter, Injustice Gods Among Us Ultimate Edition, To the Moon,
Nearest to Half-Life 2: Half-Life 2 Lost Coast, Portal, Half-Life 2 Episode Two, Left 4 Dead 2, Half-Life 2 Deathmatch, Alien Swarm, Sniper Elite V2, Abducted,
Nearest to Orcs Must Die!: Ricochet, Football Manager 2011, Proteus, Left 4 Dead 2, Half-Life 2 Lost Coast, The Elder Scrolls V Skyrim, RADical ROACH Deluxe Edition, Age of Chivalry,
Nearest to Space Pirates and Zombies: World of Goo, One Way Heroics, Lost Planet 3, Natural Selection 2, Mark of the Ninja, Endless Legend, Synergy, Kingdom Wars 2 Battles,
Nearest to FINAL FANTASY VII: Blockstorm, Trine 2, Fight The Dragon, Numba Deluxe, Nightmares from the Deep 2 The Siren`s Call, Hack, Slash, Loot, Sid Meier's Civilization V, Brawlhalla,
Nearest to Saints Row IV: The Elder Scrolls V Skyrim, Skyborn, Call of Duty Modern Warfare 2, Euro Truck Simulator 2, Divekick, Dota 2, Borderlands, Tomb Raider,
Nearest to Fallout 3 - Game of the Year Edition: Counter-Strike, Borderlands 2, Call of Duty Black Ops, Sid Meier's Civilization V, Half-Life 2 Episode One, Deus Ex Human Revolution, Fallout New Vegas, Call of Duty 4 Modern Warfare,
Nearest to Dragon Age Origins - Ultimate Edition: Portal, Trine 2, Dungeon Defenders, The Guild II - Pirates of the European Seas, Half-Life 2, English Country Tune, The Wonderful End of the World, Sid Meier's Civilization V,
Nearest to PROTOTYPE 2: Watchmen The End Is Nigh Part 2, FIFA Manager 09, FINAL FANTASY XI, Rocket League, Sid Meier's Civilization V, F1 2013, Delta Force Black Hawk Down, Eldritch,
Average loss at step 72000 : 4.73599131392
Average loss at step 74000 : 4.75961158764
Average loss at step 76000 : 4.72977783997
Average loss at step 78000 : 4.74504403515
Average loss at step 80000 : 4.63175743622
Nearest to Company of Heroes: Assassin's Creed, Company of Heroes Tales of Valor, Post Mortem, The Ball, Legend of Mysteria, TransOcean The Shipping Company, S.K.I.L.L. - Special Force 2, Into the Dark,
Nearest to Deus Ex Human Revolution: Call of Duty Black Ops - Multiplayer, Darksiders, Fallout New Vegas, Call of Duty Modern Warfare 3, Psycho Starship Rampage, Obscure 2, Wargame European Escalation, Red Faction Armageddon,
Nearest to Mirror's Edge: A Valley Without Wind 2, The Cat Lady, The Nightmare Cooperative, Team Fortress 2, Fingered, Rampage Knights, Chivalry Medieval Warfare, Trine,
Nearest to Counter-Strike Global Offensive: Team Fortress 2, Grand Theft Auto IV, Garry's Mod, Dota 2, Counter-Strike, Dirty Bomb, Trove, Abducted,
Nearest to Far Cry 3: METAL GEAR SOLID V GROUND ZEROES, Operation Flashpoint Dragon Rising, Citadels, Take Command Second Manassas, Enemy Front, MLB 2K11, Halo Spartan Assault, Sid Meier's Civilization V,
Nearest to Team Fortress 2: Dota 2, Counter-Strike Global Offensive, Unturned, Garry's Mod, Counter-Strike Source, War Thunder, Dirty Bomb, Call of Duty Modern Warfare 3 - Multiplayer,
Nearest to Fallout 4: MURDERED SOUL SUSPECT, Sid Meier's Civilization V, Darkest Dungeon, Age of Empires III Complete Collection, Coin Crypt, The Cat Lady, Dragon Age Origins - Ultimate Edition, StarDrive 2,
Nearest to Batman Arkham Asylum GOTY Edition: The Binding of Isaac, Metro 2033, Saints Row 2, THE KING OF FIGHTERS XIII STEAM EDITION, Disney Infinity 3.0 Play Without Limits, Worms Reloaded, To the Moon, Incredipede,
Nearest to Half-Life 2: Half-Life 2 Deathmatch, Half-Life 2 Episode Two, Portal, Portal 2, Day of Defeat Source, Counter-Strike Global Offensive, Counter-Strike Source, Counter-Strike,
Nearest to Orcs Must Die!: Proteus, Left 4 Dead 2, Armada 2526, Worms Armageddon, Enigmatis 2 The Mists of Ravenwood, Gods Will Be Watching, Blacklight Retribution, Killing Floor,
Nearest to Space Pirates and Zombies: One Way Heroics, Abyss Odyssey, Endless Legend, Original War, Lost Planet 3, Kingdom Wars 2 Battles, Airport Madness World Edition, Ben There, Dan That!,
Nearest to FINAL FANTASY VII: The Raven - Legacy of a Master Thief, Bientt l't, Oscura Lost Light, Shoot Many Robots, Audiosurf, Deus Ex Human Revolution, Defense Grid 2 A Matter of Endurance, Alice Madness Returns,
Nearest to Saints Row IV: Garry's Mod, Grand Theft Auto V, Borderlands 2, Clicker Heroes, SpeedRunners, Robocraft, Toribash, Left 4 Dead 2,
Nearest to Fallout 3 - Game of the Year Edition: Borderlands 2, The Elder Scrolls V Skyrim, Terraria, Dungeonbowl Knockout Edition, Torchlight II, Left 4 Dead 2, Grand Theft Auto V, Counter-Strike Source,
Nearest to Dragon Age Origins - Ultimate Edition: Adventure Park, Alien Swarm, Ori and the Blind Forest, BioShock, Battlestations Midway, Trine 2, Path of Exile, Fable - The Lost Chapters,
Nearest to PROTOTYPE 2: Arma 2, White Noise Online, Eldritch, Commandos 2 Men of Courage, F1 2013, Watchmen The End Is Nigh Part 2, The Clockwork Man, Stellar 2D,
Average loss at step 82000 : 4.64234270844
Average loss at step 84000 : 4.5202592012
Average loss at step 86000 : 4.51721813962
Average loss at step 88000 : 4.36882975928
Average loss at step 90000 : 4.51606586085
Nearest to Company of Heroes: Warhammer 40,000 Dawn of War II  Retribution, Alien Isolation, The Dark Eye Chains of Satinav, Awesomenauts, Dungeons of Dredmor, LEGO Batman 3 Beyond Gotham, Half-Life 2 Episode One, Keen Dreams,
Nearest to Deus Ex Human Revolution: Obscure 2, Call of Duty Black Ops II, Bastion, Hospital Tycoon, Half-Life 2 Episode One, Homefront, Tom Clancy's Ghost Recon Phantoms - EU, BioShock,
Nearest to Mirror's Edge: Team Fortress 2, Rogue Legacy, Unturned, Rise of Incarnates, Trine 2, RaceRoom Racing Experience , Jagged Alliance 2 - Wildfire , Loadout,
Nearest to Counter-Strike Global Offensive: Dota 2, Trove, Garry's Mod, Half-Life 2 Deathmatch, Robocraft, Warframe, Counter-Strike Source, Team Fortress 2,
Nearest to Far Cry 3: Subnautica, The Inner World, Horizon Shift, Close Combat - Gateway to Caen, Fearless Fantasy, BioShock, Wasteland 2, Industry Giant 2,
Nearest to Team Fortress 2: Dota 2, Warframe, Garry's Mod, No More Room in Hell, Unturned, Counter-Strike Source, Counter-Strike Global Offensive, Alien Swarm,
Nearest to Fallout 4: Dying Light, TERA, Tomb Raider, Infestation Survivor Stories, Counter-Strike Global Offensive, The Forest, The Cat Lady, Call of Duty Modern Warfare 3,
Nearest to Batman Arkham Asylum GOTY Edition: Assassin's Creed II, Dead Island Epidemic, Psycho Starship Rampage, Left 4 Dead 2, Dive to the Titanic, Wallace & Gromit Ep 3 Muzzled!, Why So Evil 2 Dystopia, L.A. Noire,
Nearest to Half-Life 2: Dota 2, Half-Life 2 Episode One, Portal, Half-Life 2 Deathmatch, Half-Life 2 Episode Two, Day of Defeat Source, Legend of Dungeon, Grand Theft Auto IV,
Nearest to Orcs Must Die!: Dragons and Titans, Armada 2526, Dungeon Siege 2, Proteus, Pressure, Company of Heroes Opposing Fronts, Worms Armageddon, Akuatica,
Nearest to Space Pirates and Zombies: The Witcher 2 Assassins of Kings Enhanced Edition, Halfway, Killing Floor, Hyperdimension Neptunia Re;Birth1, This War of Mine, Endless Legend, Airport Madness World Edition, Original War,
Nearest to FINAL FANTASY VII: Aura Kingdom, TowerFall Ascension, Circuits, Tales from the Borderlands, Dark Arcana The Carnival, Aggression Europe Under Fire, Prince of Persia, Retro City Rampage DX,
Nearest to Saints Row IV: Garry's Mod, Creativerse, Grand Theft Auto V, Styx Master of Shadows, Euro Truck Simulator 2, Deadpool, Block N Load, Dota 2,
Nearest to Fallout 3 - Game of the Year Edition: Borderlands The Pre-Sequel, Space Colony, Marlow Briggs, Fallout, Dungeonbowl Knockout Edition, Thief Deadly Shadows, Gurumin A Monstrous Adventure, Pillars of Eternity,
Nearest to Dragon Age Origins - Ultimate Edition: Fable - The Lost Chapters, Portal, Sid Meier's Civilization V, DARK SOULS II, Sanctum 2, Adventure Park, RAW - Realms of Ancient War, Far Cry 3 Blood Dragon,
Nearest to PROTOTYPE 2: White Noise Online, Rock of Ages, S.K.I.L.L. - Special Force 2, Star Crusade CCG, Blackguards, Anomaly Korea, Subnautica, ONE PIECE PIRATE WARRIORS 3,
Average loss at step 92000 : 4.22607941908
Average loss at step 94000 : 4.34663830309
Average loss at step 96000 : 4.25896469482
Average loss at step 98000 : 4.33921565647
Average loss at step 100000 : 4.32336386931
Nearest to Company of Heroes: Company of Heroes Tales of Valor, Call of Duty Modern Warfare 2, Company of Heroes (New Steam Version), Warhammer 40,000 Dawn of War II  Retribution, RWBY Grimm Eclipse, Warhammer 40,000 Dawn of War II, Warlock - Master of the Arcane, Burnout Paradise The Ultimate Box,
Nearest to Deus Ex Human Revolution: RPG Maker 2000, Midnight Mysteries Salem Witch Trials, aerofly RC 7, Constant C, Star Wolves 3 Civil War, Neverwinter Nights 2 Platinum, Conflict Denied Ops, Call of Duty Modern Warfare 3,
Nearest to Mirror's Edge: Team Fortress 2, PAYDAY 2, Call of Duty Black Ops II - Multiplayer, Unturned, Call of Duty Modern Warfare 3, Call of Duty World at War, Slender The Arrival, Need for Speed SHIFT,
Nearest to Counter-Strike Global Offensive: Rocket League, Trove, Garry's Mod, Dota 2, Team Fortress 2, Arma 2 Operation Arrowhead, Loadout, Unturned,
Nearest to Far Cry 3: DiRT 3, Sniper Elite V2, Portal, BioShock Infinite, DC Universe Online, Portal 2, Run and Fire, Dota 2,
Nearest to Team Fortress 2: Counter-Strike Source, Unturned, Trove, Garry's Mod, Counter-Strike Global Offensive, Dota 2, Robocraft, No More Room in Hell,
Nearest to Fallout 4: Obscure 2, The Elder Scrolls V Skyrim, Grand Theft Auto V, The Forest, Dying Light, Call of Duty World at War, FINAL FANTASY XI, Call of Duty Black Ops,
Nearest to Batman Arkham Asylum GOTY Edition: SanctuaryRPG Black Edition, Batman Arkham City GOTY, BioShock, Spec Ops The Line, Assassin's Creed II, Half-Life, Let the Cat In, BioShock 2,
Nearest to Half-Life 2: Half-Life 2 Episode Two, Half-Life 2 Episode One, Half-Life 2 Deathmatch, Portal 2, Day of Defeat Source, Half-Life, Half-Life 2 Lost Coast, Portal,
Nearest to Orcs Must Die!: Worms Armageddon, Blacklight Retribution, Tree of Life, Call of Juarez The Cartel, Mafia II, Wizardry Online, Resident Evil 5 / Biohazard 5, Borderlands 2 RU,
Nearest to Space Pirates and Zombies: Halfway, Abducted, Torchlight II, Killing Floor, Guilty Gear Isuka, Hyperdimension Neptunia Re;Birth1, Airport Madness World Edition, Abyss Odyssey,
Nearest to FINAL FANTASY VII: Dark Arcana The Carnival, War of Beach, Divinity Original Sin, Uncrowded, The Flame in the Flood, Saints Row The Third, Fallen Earth, FEZ,
Nearest to Saints Row IV: Dota 2, Akuatica, Styx Master of Shadows, Grand Theft Auto V, Borderlands 2, Deadpool, Counter-Strike Nexon Zombies, DC Universe Online,
Nearest to Fallout 3 - Game of the Year Edition: ibb & obb, Space Hulk Ascension, Torchlight II, Half-Life 2 Episode Two, Trine 2, Ghostbusters Sanctum of Slime, You Need A Budget 4 (YNAB), Fable - The Lost Chapters,
Nearest to Dragon Age Origins - Ultimate Edition: Thief 2, Grand Theft Auto San Andreas, The Guild II - Pirates of the European Seas, Hacknet, XCOM Enemy Unknown, Sid Meier's Starships, MURDERED SOUL SUSPECT, Europa Universalis IV,
Nearest to PROTOTYPE 2: White Noise Online, Anomaly Korea, Rock of Ages, GRID 2, Blackguards, Assassin's Creed Brotherhood, Alien Rage - Unlimited, Shift 2 Unleashed,

Data saved to embeddings_cbow_weighted.npy
```
