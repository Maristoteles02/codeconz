# Lighthouses AI Contest "Reloaded"

Lighthouses AI Contest is a turn based game built by [Hector Martin aka "marcan"](https://github.com/marcan/lighthouses_aicontest), as the challenge for the AI contest within one of the largest and oldest demoparty and LAN party, the [Euskal Encounter](https://ee32.euskalencounter.org/) in Bilbao, Spain.

I have updated from his original Python 2.7 code, to a Go 1.22 languaje and also I have added a few more changes:

- Original version
  - Python 2.7
  - PyGame for visualization
  - Based on comunications via stdin / stdout
  - Maps esigned for a few players at the same game
  - Built to run in a stand alone computer
- My "Reloaded" version
  - Go 1.22
  - No Visualization layer
  - Based on gRPC/Protobuf communication
  - Focused on allowing many players in the same game
  - Container first approach, game server is a container, each bot is a container (no matter which language)

The following command will build and start the game server + 3 bots and will show you the game progress until one of the bot wins, then all created resources (_docker containers, images, networks_) will be erased.

```bash
make docker-test
```

If you want to test more bots, change the `GAME_BOTS` variable in `Makefile` and retry.

> In case the test crashes, you can clean docker leftovers by running:
>
> ```bash
> make docker-destroy
> ```

## Test with local Go/python

If you want to test the game server and bots locally, you can run the following commands:

- `make rungs`: Run the game server
- `make runbotgo`: Run the Go bot (you need to have Go installed)
- `make runbotpy`: Run the Python bot (you need to have python installed and
  libraries from requirements.txt installed. You can create a virtualenv or use
  poetry).
