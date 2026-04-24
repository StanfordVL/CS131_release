# Project 2 Guidelines 

CS131, Spring 2026

This project asks you to choose a concept from the selection below and expand on it creatively. You should choose a topic that seems most interesting to you. (This is a great starting point as exploration for a final project as well!) 

Please see [Project 2 Guidelines](https://docs.google.com/document/d/1_w5RefFChBW3IdvhiCvtLgPGkIDNeWNKhVrLjKPL2Fw/edit?usp=sharing) for the most updated project guidelines.

## Main Notebook (50%): 

Read through the notebooks to get a sense of which one might interest you the most. Select **one** of the following topics to complete the notebook:

- Option A: 3D Transformations and Camera Calibration

- Option B: Keypoint Matching, Panorama Stitching, and Epipolar Geometry

## Exploration Notebook (50%): 

The exploration notebook is a separate later submission that expands on the concepts from a notebook of choice. 

While this section is open-ended in the topic that you choose to explore, we ask that you complete the following: 

- Review of current methods (5%) 
  - Once you’ve selected a topic or project idea, explore the literature space. Has there been academic research on this topic? Are there tutorials online, software packages, or libraries? 

  - Select **at least** 5 resources (youtube videos, papers, tutorials, opensource software, libraries, etc) and provide a short description (2-3 sentences) 

- Code (35%) 
  - We expect you to write code for this project (CS131 is, after all, a CS class 🙂). You may implement algorithms from scratch or expand on algorithms from this notebook if you would like, but using other libraries or other open-source software in a creative way is also sufficient. (ie. combining methods from different libraries). Here are some suggestions for starting points: 
    - Option A - 
      - Projection: use your calibrated intrinsics to overlay wireframe 3D shapes (for example, cubes/cylinders/spheres) on your own images.

      - Extrinsic calibration: estimate camera/object pose from known geometry, then verify it by projecting a wireframe back onto the image.

      - Vanishing points in the wild: apply vanishing-point calibration to your own photos to estimate focal length or measure scene geometry.

    - Option B - 
      - Stereo view synthesis: start from a dense disparity map and render a novel shifted-camera view of the scene.

      - Metric depth and 3D lifting: convert disparity to depth using baseline and focal length, then recover 3D points using intrinsics.

      - Virtual camera motion and rendering: apply a rigid camera transform (rotation/translation), reproject points, and handle holes/visibility in the synthesized image.

  - Consult a TA if you’re unsure or confused on what to do!

- Writeup (10%) 
  - An explanation of what you did, and how it relates to the topic of choice. Please attach any images, figures, etc. (\~200 words) 

## Submission Instructions: 

Please find the relevant Gradescope assignment for the option that you completed and submit that option only. You should submit 2 total Gradescope assignments: 

- Option \[X] Notebook Code 

- Option \[X] Written Component 
