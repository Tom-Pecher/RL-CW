# Reinforcement Learning - Bipedal Walker

<video src='https://github.com/user-attachments/assets/c8108b63-52f4-4a51-89ff-9f501e97cdca' width=180/>

- [Getting started](#getting-started)
- [Layout](#layout)
- [Running code](#running-code)
   * [Available arguments](#available-arguments)
   * [Example](#example)
- [Developing your agent](#developing-your-agent)
- [Connecting with WandB](#connecting-with-wandb)
- [Useful git commands](#useful-git-commands)

<!-- TOC --><a name="getting-started"></a>
# Getting Started

Install the project and install dependencies
```bash
# Clone repo
git clone git@github.com:Tom-Pecher/RL-CW
cd RL-CW

# Create and source the virtual environment
python -m venv venv
source venv/bin/activate # Linux
venv\Scripts\activate.bat # Windows

# Install requirements
pip install -r requirements.txt
```

<!-- TOC --><a name="layout"></a>
# Layout

The files we are working on are in the `src` directory. This contains sub-directories for everything we will be working on. These sub directories are where you should put your code.

> [!WARNING]
> Please do not modify the `main.py` file other than to add your own agent: it could be difficult to merge changes.

For example, the `src/agents/ddpg` directory contains the code for the DDPG agent. You do not need to follow this exactly; but it can serve as an example.

> [!NOTE]
> The `common` directory contains code that is shared between all agents. You can add any code you want to this directory, but it should be generic enough to be used by all agents.

```bash
src
├── agents
│   ├── common
│   │   ├── __init__.py  # Contains any code that should be run when importing the common module
│   │   ├── actor.py
│   │   ├── critic.py
│   │   └── replay_buffer.py
│   └── ddpg
│        ├── __init__.py  # Contains any code that should be run before the agent is initialized (populating caches, etc.)
│        ├── agent.py # Contains the code for the agent itself
│        ├── record.py # Contains the code for recording the agent
│        └── train.py # Contains the code for the training script (train_agent())
```

<!-- TOC --><a name="running-code"></a>
# Running code

```bash
# Navigate to the src directory
cd src

# Run the training script
python main.py

# To run an agent you can use the following command
python main.py --agent [agent-name]
```

<!-- TOC --><a name="available-arguments"></a>
## Available arguments

| Argument | Description |
| --- | --- |
| --agent | The agent to train. This is required. |
| --hardcore | Whether to use the hardcore version of the environment. |
| --render | Whether to render the environment. (note: periodic videos should still be recorded).|

<!-- TOC --><a name="example"></a>
## Example

To run the example agent, run the following command:
```bash
python main.py --agent example
```

<!-- TOC --><a name="developing-your-agent"></a>
# Developing your agent

Each agent should be in a separate directory. The directory contains a `train.py` file that contains the training code for your agent.
You can also add any other files you need for your agent, but these should be in the agent's directory.

As for what the training code should look like, you can look at the `example.py` file in the `src` directory, or either of the `dqn` and `ddpg` examples.
These examples will both save progress at intervals of 100.

<!-- TOC --><a name="connecting-with-wandb"></a>
# Connecting with WandB
The example agents use Weights and Biases ([WandB](https://wandb.ai/home)) to track progress and back up all videos and data.

WandB should be installed if you have created your virtual environment and installed everything in `requirements.txt` (see [Getting started](#getting-started))

To setup wandb, firstly sign up for an account on https://wandb.ai/site:

Then go to the [authorize page](https://wandb.ai/authorize) to create an API key. You can then leave this page open, and run the following command in a terminal (in your virtual environment):
```bash
wandb login
```

Paste your API key in here.

The runs should now all be backed up on WandB.

<!-- TOC --><a name="useful-git-commands"></a>
# Useful git commands

To push the project to git, you should follow a structure similar to this:

```bash
git checkout -b [branch-type]/[branch-name]
git commit [files]
git push -u origin [branch-type]/[branch-name]
```

The branch types should describe what this new branch does. For example:
```bash
git checkout -b bugfix/... # Fixed bug (properly)
git checkout -b hotfix/... # Fixed bug (temporary fix)
git checkout -b feature/... # New feature
git checkout -b isready/... # This particular part is ready
git checkout -b release/... # WHOLE PROJECT is ready
```

The branch names should be a more in depth description of what you have changed. Do not use uppercase, and all spaces should be hyphens `-`. For example:

```bash
git checkout -b bugfix/dqn-fixed-indexing
git checkout -b hotfix/removed-gpu-using-cpu
git checkout -b feature/added-initial-dqn
git checkout -b isready/dqn-is-ready
git checkout -b release/v1.0.0 # Only do this when ALL parts are ready
```

After creating your branch, checkout your files. You can either run `git checkout` on each individual filename, or if you have multiple files and would like to commit all of them, you can navigate to the `src` dir and run `git checkout .`

```bash
git add .
```

Once this has been done, you should push your changes.

```bash
git push -u origin [branch-name]

# for example
git push -u origin bugfix/dqn-fixed-indexing
```

After this is done, you can go to github and create a new pull request with your branch. Make sure to tag someone else in to review it!

[View Pull Requests](https://github.com/Tom-Pecher/RL-CW/pulls)
