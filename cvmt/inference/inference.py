""" Module for hosting functionalities needed for inference using rules defined by
McNamara et. al. in the paper

The cervical vertebral maturation method: A user's guide
https://pubmed.ncbi.nlm.nih.gov/29337631/
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import *
import torch
from cvmt.ml.models import load_model
from cvmt.ml.trainer import SingletaskTraining, max_indices_4d_tensor
from cvmt.ml.utils import TransformsMapping
from torchvision import transforms
from pathlib import Path
import os
from easydict import EasyDict
from matplotlib import gridspec
import time
from cvmt.ml.utils import download_wandb_model_checkpoint
from PIL import Image


def rescale_landmarks(
    landmarks: List[Tuple[int]],
    original_size: Tuple[int],
    input_size: Tuple[int] = (256, 256),
) -> np.ndarray:
    """
    Rescale landmarks to original image size.

    Parameters:
    landmarks (list): List of tuples representing landmarks in the format (x, y).
    original_size (tuple): Original size of the image in the format (height, width).
    model_size (tuple): Size of the image used by the model for prediction in the format (height, width).

    Returns:
    list: List of tuples representing rescaled landmarks.
    """
    height_ratio = original_size[0] / input_size[0]
    width_ratio = original_size[1] / input_size[1]

    rescaled_landmarks = [( y*height_ratio, x*width_ratio,) for y, x in landmarks]
    rescaled_landmarks = np.around(rescaled_landmarks, 1)
    return rescaled_landmarks


def img_coord_2_cartesian_coord(landmarks: np.ndarray) -> np.ndarray:
    """Image coordinates are defined with a reverted y (height) axis. Here
    it is changed so y axis has proper direction. Also, x and y axes order are 
    swapped, so x comes first and then y comes.
    """
    # swap height and width
    landmarks = np.flip(landmarks, 1)
    # invert the y axis as the original y axis in an image is inverted
    landmarks[:, 1] = -1 * landmarks[:, 1]
    return landmarks


def translate_landmarks(
    landmarks: Union[List, Tuple, np.ndarray],
    ref_index: int,
) -> np.ndarray:
    """Shifting all the landmarks coordinates such that the landmark at the
    ref_index is regarded as the origin of the cartesian plane.
    """
    # Ensure landmarks is a numpy array
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array(landmarks)
    # Get the reference point
    ref_point = landmarks[ref_index]
    # Translate all points such that the reference point is at the origin
    translated_landmarks = landmarks - ref_point
    return translated_landmarks


def plot_landmarks(landmarks: np.ndarray):
    """Debugging plots for simply visualizing the shape and the location
    of all landmarks.
    """
    # Ensure landmarks is a numpy array
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array(landmarks)
    
    # Separate the heights and widths into separate arrays for plotting
    heights = landmarks[:, 1]
    widths = landmarks[:, 0]
    # Create the scatter plot
    fig, ax = plt.subplots()   
    plt.scatter(widths, heights)
    # Add labels and title
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Landmarks Scatter Plot')
    plt.axis('equal')
    
    # Show the plot
    plt.show()

    
def rotate_landmarks(landmarks: np.ndarray, ref_index: int) -> np.ndarray:
    """Rotate the translated landmarks such that the two lowest points are both placed 
    at the x axis (y=0).
    """
    # Ensure landmarks is a numpy array
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array(landmarks)
    # Get the reference point
    ref_point = landmarks[ref_index]
    # Calculate the angle between the reference point and the x-axis
    angle = np.arctan2(ref_point[1], ref_point[0])
    angle = -1*angle
    # Create a rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    # Rotate all points
    rotated_landmarks = np.dot(landmarks, rotation_matrix.T)
    return rotated_landmarks


def compute_c34_features(landmarks: np.ndarray) -> Dict[str, float]:
    """Measure some preliminary length and height features from C3 and C4
    vertebrae only.
    """
    if len(landmarks) != 5:
        raise ValueError("C3 or C4 are only accepted here!")
    output = {}
    output['posterior_len'] = np.linalg.norm(landmarks[0] - landmarks[1])
    output['anterior_len'] = np.linalg.norm(landmarks[4] - landmarks[3])
    output['inferior_len'] = np.linalg.norm(landmarks[1] - landmarks[3])
    output['superior_len'] = np.linalg.norm(landmarks[0] - landmarks[4])
    output['concavity'] = landmarks[2, 1]
    return output


def compute_c2_features(landmarks: np.ndarray):
    """Measure concavity from lower border of C2 vertebrate only.
    """
    if len(landmarks) != 3:
        raise ValueError("C2 is only accepted here!")
    output = {}
    output['inferior_len'] = np.linalg.norm(landmarks[0] - landmarks[2])
    output['concavity'] = landmarks[1, 1]
    return output


def compute_c2_features_secondary(
    features: Dict[str, float],
    concavity_thresh: float, # 1.0 millimiter
) -> bool:
    """Compute the secondary features from C2 needed for applying 
    McNamara et. al. rules.
    """
    concavity = features['concavity'] >= concavity_thresh
    return concavity


def compute_c34_features_secondary(
    features: Dict[str, float],
    concavity_thresh: float, # 1.0 millimiter
    ant_pos_thresh: float, # anterior/posterior length ratio
    sup_inf_thresh: float, # superior/inferior length ratio
    rect_thresh_min: float, # horizontal to vertical ratio min
    rect_thresh_max: float, # horizontal to vertical ratio max
) -> Dict[str, bool]:
    """Compute the secondary features from C3 and C4 needed for applying 
    McNamara et. al. rules.
    """
    # initial outputs
    concavity, t, s, vr, hr = False, False, False, False, False
    # shape rules
    concavity = features['concavity'] >= concavity_thresh
    # # ratios of different borders
    ant_pos_ratio = features['anterior_len']/features['posterior_len']
    sup_inf_ratio = features['superior_len']/features['inferior_len']
    w_to_h_ratio = features['superior_len']/features['posterior_len']
    # shape of c3 and c4: either trapezoid (t), square (s), horizontal rectangle (hr), or vartical recatangle (vr)
    if ant_pos_ratio >= ant_pos_thresh and sup_inf_ratio >= sup_inf_thresh:
        # this can be either a rectangle or square
        if  w_to_h_ratio < rect_thresh_min:
            # this is a vertical rectangle
            vr = True
        elif w_to_h_ratio >= rect_thresh_min and w_to_h_ratio <= rect_thresh_max:
            # this is a square
            s = True
        elif w_to_h_ratio > rect_thresh_max:
            # this is a horizontal rectangle
            hr = True
    else:
        # this is a trapezoid like shape
        t = True
    return {'concavity': concavity, 't': t, 's': s, 'vr': vr, 'hr': hr, 
            'ant_pos_ratio': ant_pos_ratio, 'sup_inf_ratio': sup_inf_ratio,
            'w_to_h_ratio': w_to_h_ratio}


def clasify_by_rules(
    c2_conc: float,
    c3_feats_secondary: Dict[str, bool],
    c4_feats_secondary: Dict[str, bool],
    rect_thresh_min: float,
    rect_thresh_max: float,
) -> str:
    """The final classification function that takes in the features and applies
    several conditions in order to obtain the final bone age maturity stage.
    """
    # unpack features
    (c3_conc, c3_t, c3_s, c3_vr, c3_hr, c3_apr, c3_sir, c3_whr) = c3_feats_secondary.values()
    (c4_conc, c4_t, c4_s, c4_vr, c4_hr, c4_apr, c4_sir, c4_whr) = c4_feats_secondary.values()
    # fix concavity of c2: This is a rather hacking, but, necessary any ways!
    # it says, if c3 or c4 is concave, then biologically, c2 must be concave, too!
    if (c3_conc or c4_conc):
        c2_conc = True
    # with the same analogy, if c4 is concave, c3 must be concave, too!
    if c4_conc:
        c3_conc = True
    # apply rules
    stage = "undefined"
    if not(c2_conc or c3_conc or c4_conc) and c3_t and c4_t:
        stage = 'cs1'
    elif c2_conc and not(c3_conc or c4_conc) and c3_t and c4_t:
        stage = 'cs2'
    elif c2_conc and c3_conc and not(c4_conc) and c3_t and (c4_t or c4_hr):
        stage = 'cs3'
    elif c2_conc and c3_conc and c4_conc:
        if c3_whr > rect_thresh_max or c4_whr > rect_thresh_max:
            stage = 'cs4'
        elif (c3_whr >= rect_thresh_min and c3_whr <= rect_thresh_max) or (c4_whr >= rect_thresh_min and c4_whr <= rect_thresh_max):
            stage = 'cs5'
        elif c3_whr < rect_thresh_min or c4_whr < rect_thresh_min:
            stage = 'cs6'
        else:
            stage = "cs4-6"
    return stage


def classify_by_mcnamara_and_franchi(
    landmarks: np.ndarray,
    pixel_to_cm_factor: float,
    concavity_thresh: float = 1.0, # 1.0 millimiter
    ant_pos_thresh: float = 0.95, # anterior/posterior length ratio
    sup_inf_thresh: float = 0.95, # superior/inferior length ratio
    rect_thresh_min: float = 0.95, # horizontal to vertical ratio min
    rect_thresh_max: float = 1.0, # horizontal to vertical ratio max
) -> str:
    """The pipeline function that takes in the translated and rotated landmarks from all
    vertebrae, and spits out the bone age maturity stage.

    - isolate the landmarks [0,1,2] , [3,4,5,6,7] , and [8,9,10,11,12] into separate arrays.
    - draw a line from the inferior points of each cervical vertebrate. Measure the angle between 
    this line and the horizontal line passing through the left most landmarks that are the reference 
    points in each veertebrate. point 0 in C2, 4 in C3, and 9 in C4. 
    - Rotate all landmarks each vertebrate to the opposite of the measured angle above.
    - Shift all the points such that reference point of each vertebrate is placed at the origin in 
    the coordinate plane.
    - Now, measure the concavity of the inferior border of all vertebrae, i.e. 
    - the y coordinate of point 1 to the origin
    - the y coordinate of point 5 to the origin
    - the y coordinate of point 10 to the origin
    the concavity starts to appear in all vertebrae gradually from CS1 to CS4.
    - Next, measure the two vertical edges' heights of C3 and C4 as well as the two horizontal edges' 
    widths. In CS1 to 3, the left edge height of C3 and C4 are longer than the right edges 
    (resembeling a trapezoid shape). The right and left heights gets equally long in CS4 to 6 
    where they are longer than widths in higher bone age maturity stages.

    Args:
        landmarks: An array of landmarks that are scaled to the input image size.
        pixel_to_cm_factor: A float number for the ratio of pixels to one centimeter.
        concavity_thresh: The threshold over which we consider a vertebrate concave,
            1.0 millimiter is the default nominal value suggest by Uni. of Isf. dentists.
        ant_pos_thresh: The anterior/posterior length ratio threshold needed to classify the symmetry of 
            anterior to posterior borders.
        sup_inf_thresh: The superior/inferior length ratio threshold needed to classify the symmetry of 
            superior to inferior borders.
        rect_thresh_min: The lower bound using which we determine if a rectangle is horizontal or vertical.
            This ratio sets the minimum accepted value for a square, below which we have a vertical rectangle.
        rect_thresh_max: The lower bound using which we determine if a rectangle is horizontal or vertical.
            This ratio sets the maximum accepted value for a square, above which we have a horizontal rectangle.
    
    returns:
        bone age maturity stage
    """
    pixel_to_mm_factor = np.around(pixel_to_cm_factor/10)
    c2, c3, c4 = landmarks[0:3], landmarks[3:8], landmarks[8:]
    # rotating and changing the unit from pixel to millimiter landmarks
    c2_trns_rot, c3_trns_rot, c4_trns_rot = post_process_vertebral_landmarks(
        c2, c3, c4, pixel_to_mm_factor,
    )
    # preliminary features
    c2_feats = compute_c2_features(c2_trns_rot)
    c3_feats = compute_c34_features(c3_trns_rot)
    c4_feats = compute_c34_features(c4_trns_rot)
    # secondary features
    # c2
    c2_conc = compute_c2_features_secondary(c2_feats, concavity_thresh=concavity_thresh)
    # c3
    c3_feats_secondary = compute_c34_features_secondary(
        c3_feats,
        concavity_thresh, 
        ant_pos_thresh,
        sup_inf_thresh, 
        rect_thresh_min,
        rect_thresh_max,
    )
    # c4
    c4_feats_secondary = compute_c34_features_secondary(
        c4_feats,
        concavity_thresh, 
        ant_pos_thresh,
        sup_inf_thresh, 
        rect_thresh_min,
        rect_thresh_max,
    )
    # classify by rules
    stage = clasify_by_rules(
        c2_conc,
        c3_feats_secondary,
        c4_feats_secondary,
        rect_thresh_min=rect_thresh_min,
        rect_thresh_max=rect_thresh_max,
    )
    return stage


def post_process_vertebral_landmarks(
    c2: np.ndarray,
    c3: np.ndarray,
    c4: np.ndarray,
    pixel_to_mm_factor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate, translate, and rescale to millimeter the vertebral landmarks.
    """
    c2_cart = img_coord_2_cartesian_coord(c2)
    c2_trns = translate_landmarks(c2_cart, ref_index=0)
    c2_trns_rot = rotate_landmarks(c2_trns, ref_index=2)
    c2_trns_rot = np.divide(c2_trns_rot, pixel_to_mm_factor)

    c3_cart = img_coord_2_cartesian_coord(c3)
    c3_trns = translate_landmarks(c3_cart, ref_index=1)
    c3_trns_rot = rotate_landmarks(c3_trns, ref_index=3)
    c3_trns_rot = np.divide(c3_trns_rot, pixel_to_mm_factor)

    c4_cart = img_coord_2_cartesian_coord(c4)
    c4_trns = translate_landmarks(c4_cart, ref_index=1)
    c4_trns_rot = rotate_landmarks(c4_trns, ref_index=3)
    c4_trns_rot = np.divide(c4_trns_rot, pixel_to_mm_factor)
    return c2_trns_rot, c3_trns_rot, c4_trns_rot


def load_pretrained_model_eval_mode(
    model_params: Dict[str, Any],
    use_pretrain: bool,
    checkpoint_path: str,
    task_id: int,
    loss_name: str,
) -> torch.nn.Module:
    """Load the model and use for inference.
    
    Args:
        model_params: The parameters of the model. This is directly passed to model loading function.
        use_pretrain: bool,
        checkpoint_path: str,

    Returns:
        pretrained model loaded.
    """
    # instantiate the bare-bone model
    model = load_model(**model_params)
    # instantiate the pytorch lightning module that was used for training
    pl_module = SingletaskTraining(
        model=model,
        task_id=task_id,
        checkpoint_path=checkpoint_path,
        loss_name=loss_name,
    )
    # check if we want to load weights from a checkpoint
    if use_pretrain:
        model = pl_module.load_from_checkpoint(checkpoint_path,).model
    # se the device to `cuda` is available, if not, cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # turn off gradient accumulation
    model.eval()
    return model, device


def predict_image(
    image: np.ndarray,
    model: torch.nn.Module,
    task_id: int,
    transforms_params: EasyDict,
    device: str = 'cuda',
) -> Union[Any, np.ndarray]:
    """Pass an image to the model and gather the outputs. The type of outputs
    in a multimodal/multitask model is determined by the parameter `task_id`.
    
    Args:
        image: input image,
        model: The loaded model,
        task_id: The task id,
        input_size: The model input size,

    Returns:
        Model outputs
    """
    # load the input image
    im_w, im_h = image.size
    # create the transforms
    transforms_mapping = TransformsMapping()
    transforms_config = OrderedDict(transforms_params)
    my_transforms = [transforms_mapping.get(t_name, **t_args) for t_name, t_args in transforms_config.items()]
    my_transforms = transforms.Compose(my_transforms)
    # preprocess the input
    input_tensor = my_transforms(image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)
    # Perform inference
    with torch.no_grad():
        model_output = model(input_batch, task_id=task_id)
    # postprocess the outputs
    output = post_process_outputs(
        task_id=task_id,
        model_output=model_output,
        original_size=(im_h, im_w,),
        input_size=transforms_params.RESIZE.size,
    )
    return output


def post_process_outputs(
    task_id: int,
    model_output: np.ndarray,
    **kwargs,
) -> Union[Any, np.ndarray]:
    """ Postprocessing function for vertebral and facial landmark detection tasks.
    """
    if task_id in [3, 4]:
        # Post-process the output
        landmarks_coords = max_indices_4d_tensor(model_output)
        landmarks_coords = landmarks_coords.squeeze()
        landmarks_coords = landmarks_coords.cpu().numpy()
        # rescale the obtained landmarks with respect to the original shape of the input image
        output = rescale_landmarks(landmarks_coords, **kwargs)
    return output


def plot_image_and_vertebral_landmarks(
    img_name: str,
    model_id: str,
    landmarks: np.ndarray,
    image: np.ndarray,
):
    """Create a grid of two and plot the same image in both of the grids. Then,
    overlay the predicted landmarks on the second image. The landmarks must be
    scaled to the image size.
    """
    fig = plt.figure(tight_layout=True, figsize=(15,15))
    gs = gridspec.GridSpec(1, 2)

    # show the image
    ax = fig.add_subplot(gs[0])
    ax.imshow(image, cmap='gray')

    ax = fig.add_subplot(gs[1])
    ax.imshow(image, cmap='gray')

    # plot the landmarks on top of the image
    ax.scatter(landmarks[:, 1], landmarks[:, 0], s=10, c='cyan')
    for i, l in enumerate(landmarks):
        ax.text(l[1], l[0], str(i), color='greenyellow')  # Annotate the index

    plt.tight_layout()
    plt.show()
    verify_dir = "artifacts/verification"
    fig.savefig(os.path.join(verify_dir, f"{img_name}_{model_id}.jpg"), dpi=300)
    time.sleep(1)


def predict_image_cmd_interface(params: Union[EasyDict, Dict], filepath: str, px2cm_ratio: float):
    """Pass on image file to the model stred at the location provided by
    `filepath` arg.
    """
    # unpack parameters
    use_pretrain = True
    task_config = params.TRAIN.V_LANDMARK_TASK
    task_id = task_config.TASK_ID
    loss_name = params.TRAIN.LOSS_NAME
    model_params = params.MODEL.PARAMS
    transforms_params = params.INFERENCE.TRANSFORMS
    mcnamara_args = params.INFERENCE.MCNAMARA.ARGS
    concavity_thresh = mcnamara_args.concavity_thresh
    ant_pos_thresh = mcnamara_args.ant_pos_thresh
    sup_inf_thresh = mcnamara_args.sup_inf_thresh
    rect_thresh_min = mcnamara_args.rect_thresh_min
    rect_thresh_max = mcnamara_args.rect_thresh_max
    # load the image
    image = Image.open(filepath)
    # load model
    checkpoint_path, model_id = download_wandb_model_checkpoint(
        wandb_checkpoint_uri= params.VERIFY.WANDB_CHECKPOINT_REFERENCE_NAME
    )
    model, device = load_pretrained_model_eval_mode(
        model_params=model_params,
        use_pretrain=use_pretrain,
        checkpoint_path=checkpoint_path,
        task_id=task_id,
        loss_name=loss_name,
    )
    # pass input image to the model
    rescaled_landmarks_coords = predict_image(
        image=image,
        model=model,
        task_id=task_id,
        transforms_params=transforms_params,
        device=device,
    )
    # get the maturity stage
    stage = classify_by_mcnamara_and_franchi(
        rescaled_landmarks_coords,
        px2cm_ratio,
        concavity_thresh,
        ant_pos_thresh,
        sup_inf_thresh,
        rect_thresh_min,
        rect_thresh_max,
    )
    print("stage is: ", stage)
    return stage
