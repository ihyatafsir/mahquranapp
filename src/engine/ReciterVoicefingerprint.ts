/**
 * Reciter Tajweed Voicefingerprint - Client Runtime Engine
 * Models reciter-specific vocal tract physical attributes, mora tempo,
 * and biomechanical articulatory mechanics.
 */

export interface ReciterVoicefingerprint {
  id: string;
  name: string;
  style: string;
  baseMoraMs: number;
  maddLazimWeight: number;
  maddWajibWeight: number;
  maddAridWeight: number;
  ghunnahWeight: number;
  shaddahWeight: number;
  maddTabieeWeight: number;
  sukoonPlosiveWeight: number;
  normalConsonantWeight: number;
  silentLetterMs: number;
  qalqalahBounceMs: number;
  organWeights: {
    Jawf: number;
    Halq: number;
    Lisan: number;
    Shafatan: number;
    Khayshum: number;
  };
}

export const RECITER_VOICEFINGERPRINTS: Record<string, ReciterVoicefingerprint> = {
  abdul_basit_murattal: {
    id: "abdul_basit_murattal",
    name: "Sheikh AbdulBaset (Murattal)",
    style: "Classical Egyptian Murattal",
    baseMoraMs: 172,
    maddLazimWeight: 10.2,
    maddWajibWeight: 6.8,
    maddAridWeight: 6.2,
    ghunnahWeight: 3.8,
    shaddahWeight: 2.9,
    maddTabieeWeight: 2.2,
    sukoonPlosiveWeight: 0.85,
    normalConsonantWeight: 1.0,
    silentLetterMs: 0.005,
    qalqalahBounceMs: 0.045,
    organWeights: {
      Jawf: 1.15,
      Halq: 1.0,
      Lisan: 0.95,
      Shafatan: 1.05,
      Khayshum: 1.2,
    },
  },
  minshawi_mujawwad: {
    id: "minshawi_mujawwad",
    name: "Sheikh Mohamed Siddiq Al-Minshawi",
    style: "Tahqeeq / Emotive Mujawwad",
    baseMoraMs: 240,
    maddLazimWeight: 11.5,
    maddWajibWeight: 7.8,
    maddAridWeight: 7.2,
    ghunnahWeight: 4.2,
    shaddahWeight: 3.4,
    maddTabieeWeight: 2.4,
    sukoonPlosiveWeight: 0.95,
    normalConsonantWeight: 1.05,
    silentLetterMs: 0.005,
    qalqalahBounceMs: 0.060,
    organWeights: {
      Jawf: 1.25,
      Halq: 1.15,
      Lisan: 1.05,
      Shafatan: 1.0,
      Khayshum: 1.3,
    },
  },
  mah: {
    id: "mah",
    name: "Sheikh Mohammad Ahmad Hassan",
    style: "Hadr / Modern Murattal",
    baseMoraMs: 125,
    maddLazimWeight: 7.5,
    maddWajibWeight: 5.2,
    maddAridWeight: 4.8,
    ghunnahWeight: 3.0,
    shaddahWeight: 2.4,
    maddTabieeWeight: 1.8,
    sukoonPlosiveWeight: 0.75,
    normalConsonantWeight: 0.95,
    silentLetterMs: 0.005,
    qalqalahBounceMs: 0.035,
    organWeights: {
      Jawf: 1.0,
      Halq: 0.95,
      Lisan: 1.0,
      Shafatan: 1.0,
      Khayshum: 1.0,
    },
  },
  abdul_basit: {
    id: "abdul_basit",
    name: "Sheikh AbdulBaset (Mujawwad)",
    style: "Monumental Grand Mujawwad",
    baseMoraMs: 260,
    maddLazimWeight: 14.0,
    maddWajibWeight: 9.0,
    maddAridWeight: 8.5,
    ghunnahWeight: 4.5,
    shaddahWeight: 3.6,
    maddTabieeWeight: 2.6,
    sukoonPlosiveWeight: 1.0,
    normalConsonantWeight: 1.1,
    silentLetterMs: 0.005,
    qalqalahBounceMs: 0.065,
    organWeights: {
      Jawf: 1.35,
      Halq: 1.2,
      Lisan: 1.0,
      Shafatan: 1.1,
      Khayshum: 1.35,
    },
  },
};

export function getVoicefingerprint(reciterId: string): ReciterVoicefingerprint {
  return RECITER_VOICEFINGERPRINTS[reciterId] || RECITER_VOICEFINGERPRINTS.abdul_basit_murattal;
}
