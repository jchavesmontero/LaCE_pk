from lace.cosmo.camb_cosmo import dkms_dMpc


def change_units(in_unit, out_unit, z, cosmo=None, camb_results=None):
    lambda_alpha = 1215.67
    clight = 299792.458
    units = ["kms", "AA", "Mpc"]
    # check in_unit and out_unit in units

    if ((in_unit == "Mpc") | (out_unit == "Mpc")) & (cosmo is None):
        raise ValueError("Please introduce cosmology")

    if (in_unit == "AA") & (out_unit == "kms"):
        k_convert = lambda_alpha * (1 + z) / clight
    elif (in_unit == "kms") & (out_unit == "AA"):
        k_convert = 1 / (lambda_alpha * (1 + z) / clight)
    elif (in_unit == "kms") & (out_unit == "Mpc"):
        k_convert = dkms_dMpc(cosmo, z, camb_results=camb_results)
    elif (in_unit == "Mpc") & (out_unit == "kms"):
        k_convert = 1 / dkms_dMpc(cosmo, z, camb_results=camb_results)
    elif (in_unit == "AA") & (out_unit == "Mpc"):
        AA2kms = lambda_alpha * (1 + z) / clight
        kms2Mpc = dkms_dMpc(cosmo, z, camb_results=camb_results)
        k_convert = AA2kms * kms2Mpc
    elif (in_unit == "Mpc") & (out_unit == "AA"):
        AA2kms = lambda_alpha * (1 + z) / clight
        kms2Mpc = dkms_dMpc(cosmo, z, camb_results=camb_results)
        k_convert = 1 / (AA2kms * kms2Mpc)
    return k_convert
