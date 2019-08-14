def denormalize(imgs, dataset):
    """Denormalize the images.
    Args:
        imgs (torch.Tensor) (N, C, H, W): Te images to be denormalized.
        dataset (str): 
        
    Returns:
        imgs (torch.Tensor) (N, C, H, W): The denormalized images.
    """
    if dataset not in ['acdc']:
        raise ValueError(f"The name of the dataset should be 'acdc'. Got {dataset}.")
    
    if dataset == 'acdc':
        mean, std = 54.089, 48.084
        
    imgs = imgs.clone()
    imgs = (imgs * std + mean).round().clamp(0, 255)
    return imgs