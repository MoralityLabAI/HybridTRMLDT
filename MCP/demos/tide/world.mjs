export const facts = {
  lens: "A map of the lanterns",
  bell: "The missing bell beat",
  seal: "A ferryman’s seal",
};

export const scenes = {
  quay: {
    title:"A harbor holding its breath", place:"Low Quay · Before slack water", art:"quay",
    text:"The ferry has gone. Across the harbor a single amber window burns in the island observatory. The keeper left you a note: where the lights disagree, listen for the beat that does not come.",
    choices:[
      {id:"archive", label:"Follow the amber lamps to the chart room", to:"archive"},
      {id:"tower", label:"Climb the bell tower", to:"tower"},
      {id:"ferry", label:"Ask the night ferryman for help", to:"ferry"},
      {id:"causeway", label:"Walk down to the drowned causeway", to:"causeway"},
    ],
  },
  archive: {
    title:"The chart room", place:"Lantern Archive", art:"archive",
    text:"Unbound charts turn in a draft. The old map marks three guiding lights; the new chart marks four. The extra light describes a reflection, drawn deliberately by a hand that knew the water.",
    choices:[
      {id:"study", label:"Trace the true lanterns and return to the quay", to:"quay", grant:"lens"},
      {id:"back", label:"Leave the charts undisturbed", to:"quay"},
    ],
  },
  tower: {
    title:"One beat is missing", place:"The Tide Bell", art:"tower",
    text:"At the third stroke you expect the stones to shake. Instead there is a pocket of silence. A strip of ribbon tied to the bell frame rises in that gap, toward the island.",
    choices:[
      {id:"listen", label:"Count the silent beat and go down", to:"quay", grant:"bell"},
      {id:"back", label:"Leave the tower", to:"quay"},
    ],
  },
  ferry: {
    title:"A borrowed seal", place:"The Ferryman’s Steps", art:"ferry",
    text:"The ferryman has tied his boat high on the steps. He will lend you his seal, but warns that the watch patrols the shortest passage from this quay to the island.",
    choices:[
      {id:"borrow", label:"Borrow the seal and return to the quay", to:"quay", grant:"seal"},
      {id:"back", label:"Return empty-handed", to:"quay"},
    ],
  },
  causeway: {
    title:"Stones under water", place:"The Drowned Causeway", art:"causeway",
    text:"Four lights seem to float above the water. Their reflections look like steps. Beyond them, a square of dry stone waits for the tide to turn.",
    choices:[
      {id:"true-crossing", label:"Cross by the true lights on the silent beat", to:"arrive", requires:["lens","bell"], lock:"Find the chart and the bell beat"},
      {id:"sealed-crossing", label:"Show the seal to the harbor watch", to:"watch", requires:["seal"], lock:"Borrow the ferryman’s seal"},
      {id:"risk", label:"Follow the brightest reflection", to:"lost"},
      {id:"back", label:"Return to the quay", to:"quay"},
    ],
  },
  arrive: {
    title:"The keeper’s window", place:"Island Observatory · Dawn", art:"arrive",
    text:"You reach the dry platform between strokes of the bell. In the observatory, the keeper moves the single lantern to the window. Behind you, the three true lights reappear on the falling tide.",
    choices:[],
  },
  watch: {
    title:"An escort through dawn", place:"Harbor Watch · Dawn", art:"watch",
    text:"The watch recognizes the seal and brings you across on a supply skiff. By the time you reach the observatory, its window is dark. You leave your note with the guard at the door.",
    choices:[],
  },
  lost: {
    title:"The light in the water", place:"Outer Flats · Night", art:"lost",
    text:"The brightest lantern doubles, then disappears. You turn back before the stone drops away. The quay is near, but the crossing closes for the night.",
    choices:[],
  },
};

export const begin = () => ({node:"quay", held:[], trail:["quay"]});

export function options(world, state) {
  return world[state.node].choices.map(choice => ({
    ...choice, available:(choice.requires ?? []).every(fact=>state.held.includes(fact)),
  }));
}

export function choose(world, state, action) {
  const choice=options(world,state).find(item=>item.id===action);
  if(!choice?.available) return state;
  const held=choice.grant && !state.held.includes(choice.grant) ? [...state.held,choice.grant] : state.held;
  return {node:choice.to,held,trail:[...state.trail,choice.to]};
}
