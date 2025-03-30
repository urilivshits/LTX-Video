# User Tasks
<!-- 
FORMAT GUIDE - DO NOT DELETE

## User Query: "[Exact query text as a single string with all line breaks removed]"
  - Task: [Brief task description]
    - [ ] Define test cases for [specific functionality]
      - [ ] Unit test: [test case description]
      - [ ] Service test: [test case description]
      - [ ] API test: [test case description]
    - [ ] Implement [specific functionality]
      - [ ] [Specific implementation subtask]
      - [ ] [Specific implementation subtask]
    - [ ] Run tests and validate implementation
      - [ ] Run unit tests for [specific functionality]
      - [ ] Run service tests for [specific functionality]
      - [ ] Run API tests for [specific functionality]
  - Task: [Brief task description]
    - [ ] Define test cases for [specific functionality]
      - [ ] Unit test: [test case description]
      - [ ] Service test: [test case description]
      - [ ] API test: [test case description]
    - [ ] Implement [specific functionality]
      - [ ] [Specific implementation subtask]
      - [ ] [Specific implementation subtask]
    - [ ] Run tests and validate implementation
      - [ ] Run unit tests for [specific functionality]
      - [ ] Run service tests for [specific functionality]
      - [ ] Run API tests for [specific functionality]
-->

## User Query: "reindex the repo, renew the artifacts, check out the new repo rules and let me know"
  - Task: Reindex the repository
    - [x] List all directories and files in the repository
    - [x] Check current state of artifact files
    - [x] Understand the project structure through README and main code
  - Task: Renew the artifacts according to repo rules
    - [x] Update the tasks.md file structure according to the cursor rules
    - [x] Update the context.md file structure according to the cursor rules
  - Task: Report on repository rules
    - [x] Review the provided cursor rules
    - [x] Summarize key aspects of the cursor rules
    - [x] Report findings to the user

## User Query: "note i removed the naming conventions from the rules"
  - Task: Acknowledge rule changes
    - [x] Review updated cursor rules
    - [x] Confirm removal of naming conventions
    - [x] Provide confirmation to user

## User Query: "based on this repo @README.md and based on this repo docs @https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx_video lets create a new workflow based on the existing workflow for @03_image_to_video_diffusers.py : 1. the new workflow must be in a new file in @USER_DIR 2. it should allow diffusers-based image to video conversions 3. it should allow expanding the video by the following logic (suggest if there is a better logic): - based on original input image generate a video - take the last frame of the generated video and save it - generate a new expanded video based on the last frame of the previous video"
  - Task: Create video expander workflow
    - [x] Define functionality for the video expander
      - [x] Define core functionality: image to video with expansion
      - [x] Define video concatenation with smooth transitions
      - [x] Define proper file management for segments and output
    - [x] Implement video expander workflow
      - [x] Create new file in USER_DIR (04_video_expander_diffusers.py)
      - [x] Implement diffusers-based image-to-video conversion
      - [x] Implement video expansion logic using last frames
      - [x] Add video segment concatenation with crossfade
      - [x] Implement command-line arguments for user configuration
    - [x] Ensure proper documentation
      - [x] Add comprehensive script header with explanation
      - [x] Document functions with clear docstrings
      - [x] Include example command line usage

## User Query: "i see we added some dependencies, lets add them to requirements of the project and lets then also install them into the venv_cuda env."
  - Task: Add new dependencies to project
    - [x] Identify new dependencies needed for the video expander workflow
      - [x] Identify opencv-python for video processing
      - [x] Identify numpy for array operations
      - [x] Identify argparse for command-line arguments
      - [x] Identify other dependencies used in the script
    - [x] Add dependencies to project requirements
      - [x] Create requirements.txt file with all needed dependencies
      - [x] Group dependencies by functionality
    - [x] Install dependencies in venv_cuda environment
      - [x] Install all dependencies from requirements.txt
      - [x] Verify successful installation through imports

## User Query: ""
  - Task: Fix bugs in video expander script
    - [x] Identify bug in the save_video_frames function
      - [x] Diagnose issue with PIL Image handling
      - [x] Determine correct approach for handling different frame types
    - [x] Implement bug fix for frame saving
      - [x] Update save_video_frames function to handle PIL Image objects
      - [x] Update save_frame_as_image function for consistency
      - [x] Fix height/width detection to use the saved frames
    - [x] Fix additional bug in last frame processing
      - [x] Diagnose issue with frame type in expansion iterations
      - [x] Update frame processing to handle both numpy arrays and PIL Images
    - [x] Document changes in tasks.md

## User Query: "cool, this works. now how do we make sure the prompt for videos n++ still makes sense, any ideas?"
  - Task: Enhance video expansion with improved prompting
    - [x] Research techniques for prompt coherence across video segments
    - [x] Design solution for temporal prompting across multiple segments
    - [x] Present options to the user for consideration

## User Query: "which seed is now used for second and third generations? maybe logic around seeds?"
  - Task: Enhance seed management across video segments
    - [x] Analyze current seed implementation in video expansion
    - [x] Research options for seed derivation in multi-segment generation
    - [x] Present seed management alternatives to the user
    - [x] Discuss hybrid approach for balancing consistency and variation

## User Query: "lets do hybrid"
  - Task: Verify hybrid seed approach implementation
    - [x] Confirm hybrid seed approach is already implemented in the script
    - [x] Explain to user how the current implementation works
    - [x] Note that no additional dependencies are needed

## User Query: "cool, it works. and how do i improve the output because if i try to make a video of a person from an image then by the end of the first video segment the person is already not maintaining the physical form and so the next segment starts with a bad input image as the result"
  - Task: Improve subject consistency across video segments
    - [x] Research techniques for maintaining subject appearance in extended videos
      - [x] Explore image regression toward original image
      - [x] Investigate frame selection alternatives
      - [x] Research conditioning scale adjustments
    - [x] Implement enhanced subject consistency features
      - [x] Add regression strength option to blend with original image
      - [x] Add middle frame selection option instead of last frame
      - [x] Implement segment-specific prompts for better temporal control
      - [x] Add conditioning scale parameter for better image adherence
    - [x] Document new parameters and functionality
      - [x] Add command-line arguments for all new options
      - [x] Update configuration output to show new parameters
      - [x] Implement image blending function for regression feature

## User Query: ""
  - Task: Fix unsupported parameter in video expander script
    - [x] Identify error with conditioning_scale parameter
      - [x] Verify the parameter is not supported by LTXImageToVideoPipeline
    - [x] Fix the error by removing unsupported parameter
      - [x] Remove conditioning_scale parameter from pipeline call
      - [x] Remove related command-line argument
      - [x] Update configuration output
    - [x] Document changes in tasks.md

## User Query: "describe for me each of this options please: Start with --regression_strength 0.35 (adjust between 0.2-0.5 as needed) Try --use_middle_frame for better transition frames Use more detailed prompts about the person's physical appearance Increase --guidance_scale to 3.5-4.0 for better prompt adherence Use --condition_scale 1.2 or higher to preserve more image details"
  - Task: Explain subject consistency parameters
    - [x] Provide detailed explanations of each parameter
      - [x] Explain regression_strength parameter function and values
      - [x] Explain use_middle_frame purpose and benefits
      - [x] Explain importance of detailed prompts
      - [x] Explain guidance_scale impact on consistency
      - [x] Clarify condition_scale parameter removal
    - [x] Document best practices for parameters
      - [x] Recommend optimal value ranges
      - [x] Explain when to adjust parameters
      - [x] Note parameter interdependencies

## User Query: "cool, lets add all the relevant new params to the top of the file together with other params for easy access"
  - Task: Reorganize parameters at top of video expander script
    - [x] Add new subject consistency parameters to USER CONFIGURABLE PARAMETERS section
      - [x] Add regression_strength with default recommended value and documentation
      - [x] Add use_middle_frame flag with default setting
      - [x] Add segment_prompts with example format
    - [x] Update command-line argument defaults to match top parameters
    - [x] Improve parameter documentation with recommended values
    - [x] Update guidance_scale to recommended value

## User Query: "yes but follow the project rules and update all the necessary docs please also please update all the prompts used in the repo to be generic prompts descibing a person"
  - Task: Update documentation according to project rules
    - [x] Update tasks.md with the new task
    - [x] Update context.md with new feature and architectural changes
  - Task: Update prompt handling in video expander workflow
    - [x] Add clearer logging for prompt usage in each segment
    - [x] Add parameter to optionally enhance all segment prompts
    - [x] Fix prompt truncation to avoid cutting words in half
    - [x] Improve documentation of prompt enhancement options
  - Task: Update all prompts in repository to be generic descriptions
    - [x] Update prompts in USER_DIR/03_image_to_video_diffusers.py
    - [x] Update prompts in USER_DIR/04_video_expander_diffusers.py
    - [x] Remove inappropriate or overly specific prompts
    - [x] Replace with generic person descriptions focused on natural movements
  - Task: Finalize documentation and file checks
    - [x] Update README.md with latest enhancement options and correct file references
    - [x] Verify .gitignore and .env files
    - [x] Mark final tasks as complete in tasks.md
    - [x] Mark final updates in context.md