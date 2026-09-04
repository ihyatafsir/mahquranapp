export interface Reciter {
  id: string;
  name: string;
  shortName: string;
  description: string;
}

export const RECITERS: Reciter[] = [
  {
    id: "abdul_basit_murattal",
    name: "Sheikh AbdulBaset AbdulSamad (Murattal)",
    shortName: "Abdul Basit (Murattal) • 114 Surahs",
    description: "Egyptian Master Reciter • Classical Murattal Style",
  },
  {
    id: "minshawi_mujawwad",
    name: "Sheikh Mohamed Siddiq Al-Minshawi",
    shortName: "Al-Minshawi (Mujawwad)",
    description: "Egyptian Master Reciter • Classical Tahqeeq Style",
  },
  {
    id: "mah",
    name: "Sheikh Mohammad Ahmad Hassan",
    shortName: "Mohammad Ahmad Hassan (MAH)",
    description: "Acoustic Alignment & Tajweed Guided Physics",
  },
  {
    id: "abdul_basit",
    name: "Sheikh AbdulBaset AbdulSamad",
    shortName: "Abdul Basit (Mujawwad)",
    description: "Egyptian Master Reciter",
  },
];
