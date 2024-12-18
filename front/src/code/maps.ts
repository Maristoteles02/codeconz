import {
  game,
  setup,
  player,
  lighthouse,
  round,
  turn,
  p,
} from "@/code/domain.js";
import { colorFor } from "@/code/palette.js";

const gridFor = (map) =>
  map.map((row) => row.split("").map((col) => (col === " " ? "w" : "g")));

const playersFor = (map) =>
  map.reduce((players, row, y) => {
    for (let x = 0; x < row.length; x++) {
      if (/\d/.test(row[x])) {
        const index = parseInt(row[x]);
        players[index - 1] = player(
          index,
          p(x, y),
          0,
          0,
          [],
          `Player ${index}`,
          colorFor(index - 1),
        );
      }
    }
    return players;
  }, []);

const lighthousesFor = (map) =>
  map.reduce((lighthouses, row, y) => {
    for (let x = 0; x < row.length; x++) {
      if (/^[a-z]$/.test(row[x])) {
        const index = row[x].charCodeAt(0) - 96;
        lighthouses[index] = lighthouse(index, 0, null, [], p(x, y));
      }
    }
    return lighthouses;
  }, []);

const energyFor = (map) =>
  map.reduce((energy, row) => {
    energy.push(row.split("").map((digit) => (parseInt(digit) || 0) * 10));
    return energy;
  }, []);

const mapFor = (tiles, rounds) =>
  game(
    gridFor(tiles),
    setup([], playersFor(tiles), lighthousesFor(tiles)),
    rounds || [],
  );

const gameFor = mapFor;

export const game_1 = gameFor(
  [
    // 23456
    "       ", //0
    " 1a..e ", //1
    " ..... ", //2
    " .d... ", //3
    " ....c ", //4
    " b.... ", //5
    "       ", //6
  ],
  [
    round(
      setup(
        [],
        [player(1, p(1, 1), 0, 0, [])],
        [lighthouse(1, 0, 1, [2], p(2, 1)), lighthouse(2, 0, 1, [1], p(1, 5))],
      ),
      [],
      0,
    ),
  ],
);

export const map_1 = mapFor([
  // 2345678901234567890
  "                     ", //0
  "  .... ...  ...    . ", //1
  " .  1.  .  . .2.. .. ", //2
  " ..  .a . .. ..... . ", //3
  " . .  .....  b...... ", //4
  " . ... .... .....3.. ", //5
  "  .... ..       . .. ", //6
  " .4.c. . ....  .  d. ", //7
  " ..   .  ..   ....   ", //8
  " .   ...  ...  ...   ", //9
  " . . ....e ... .6 .  ", //10
  "  ...  5... ..f .  . ", //11
  "  .    ...   . ...   ", //12
  " .    ....    ... .  ", //13
  "                     ", //14
]);

export const map_2 = mapFor(
  [
    // 2345678901234567890
    "                     ", //0
    " ................... ", //1
    " ................... ", //2
    " ................... ", //3
    " ................... ", //4
    " ......1............ ", //5
    " ................... ", //6
    " .........a......... ", //7
    " ................... ", //8
    " ................... ", //9
    " ................... ", //10
    " ................... ", //11
    " ................... ", //12
    " ................... ", //13
    "                     ", //14
  ],
  [
    round(
      setup(
        energyFor([
          // 2345678901234567890
          ".....................", //.
          ".....................", //1
          ".....................", //2
          "..........5..........", //3
          ".........565.........", //4
          "........56765........", //5
          ".......5678765.......", //6
          "......567898765......", //7
          ".......5778765.......", //8
          "........56765........", //9
          ".........565.........", //1.
          "..........5..........", //11
          ".....................", //12
          ".....................", //13
          ".....................", //14
        ]),
        [
          player(1, p(7, 5), 0, 0, []),
          // player(2, p(1, 2), 4, 0, [])
        ],
        [lighthouse(1, 0, null, [], p(10, 7))],
      ),
      [
        turn(player(1, p(7, 6), 0, 0, []), [
          lighthouse(1, 0, null, [], p(10, 7)),
        ]),
        // turn(player(2, p(1, 3), 4, 0, []), [
        //   lighthouse(1, 0, null, [])
        // ])
      ],
    ),

    round(
      setup(
        energyFor([
          // 2345678901234567890
          ".....................", //.
          ".....................", //1
          ".....................", //2
          "..........5..........", //3
          ".........565.........", //4
          "........56765........", //5
          ".......0678765.......", //6
          "......567898765......", //7
          ".......5778765.......", //8
          "........56765........", //9
          ".........565.........", //1.
          "..........5..........", //11
          ".....................", //12
          ".....................", //13
          ".....................", //14
        ]),
        [
          player(1, p(7, 6), 0, 0, []),
          // player(2, p(1, 2), 4, 0, [])
        ],
        [
          // lighthouse(1, 0, null, [])
        ],
      ),
      [
        // turn(player(1, p(7, 6), 0, 0, []), [
        //   lighthouse(1, 0, null, [])
        // ]),
        // turn(player(2, p(1, 3), 4, 0, []), [
        //   lighthouse(1, 0, null, [])
        // ])
      ],
    ),
  ],
);
