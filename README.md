# Reinforcement Learning - Bipedal Walker


## Getting Started

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

## Layout

The files we are working on are in the `src` directory. This contains sub-directories for everything we will be working on. These sub directories are where you should put your code.

## Running code

```bash
# Navigate to the src directory
cd src 

# Run the training script
python train.py
```

## Useful git commands

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
