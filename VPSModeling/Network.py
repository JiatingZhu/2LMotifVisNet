import numpy as np
import scipy.stats as stats
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def fun_feedforward(t_n, A, B, C, I_vec, filter_unit, sigma_Ex=1, sigma_Inh=10, amp_inh=1, amp=10, RF_scale=1,
                    amp_exc=1):
    G_Ex = stats.norm.pdf(filter_unit, loc=0, scale=RF_scale * sigma_Ex)
    G_Inh = stats.norm.pdf(filter_unit, loc=0, scale=RF_scale * sigma_Inh)

    norm_G_Ex = amp * amp_exc * G_Ex / np.sum(G_Ex) * RF_scale * sigma_Ex * np.sqrt(2 * np.pi) # * np.sum(G_Ex)

    norm_G_Inh = amp * amp_inh * G_Inh / np.sum(G_Inh) * RF_scale * sigma_Inh * np.sqrt(2 * np.pi) #* np.sum(G_Inh)
    #print('RF scale', RF_scale, ' np.sum(norm_G_Ex) = ', np.sum(G_Ex), ' np.sum(norm_G_Inh) = ', np.sum(G_Inh) )
    item_a = -A * t_n

    item_b = np.multiply((B - t_n), np.convolve(I_vec, norm_G_Ex, 'same'))
    item_c = np.multiply(-(C + t_n), np.convolve(I_vec, norm_G_Inh, 'same'))

    y = item_a + item_b + item_c
    return y, norm_G_Ex, norm_G_Inh


def fun_feedback(t_n, A, B, C, I_vec, V1, filter_unit, sigma_Ex=1, sigma_Inh=10, amp_inh=1, alpha=0.5, amp=10,
                 RF_scale=1, amp_exc=1):
    G_Ex = stats.norm.pdf(filter_unit, loc=0, scale=RF_scale * sigma_Ex)  # lgn_width_scale*2
    G_Inh = stats.norm.pdf(filter_unit, loc=0, scale=RF_scale * sigma_Inh)

    norm_G_Ex = amp * amp_exc * G_Ex / np.sum(G_Ex) * RF_scale * sigma_Ex * np.sqrt(2 * np.pi) # * np.sum(G_Ex)
    norm_G_Inh = amp * amp_inh * G_Inh / np.sum(G_Inh) * RF_scale * sigma_Inh * np.sqrt(2 * np.pi) #* np.sum(G_Inh)

    item_a = -A * t_n

    input_V1 = V1 * alpha # * 10

    item_b = np.multiply((B - t_n), np.convolve(I_vec + input_V1, norm_G_Ex, 'same'))
    item_c = np.multiply(-(C + t_n), np.convolve(I_vec + input_V1, norm_G_Inh, 'same'))
    y = item_a + item_b + item_c
    return y, norm_G_Ex, norm_G_Inh


def get_network_output_wider_V1(A,B,C,I,filter_unit, filter_unit_2, dt, timestep, pixel_size, boundary_size,
                                this_alpha=1,
                                this_amp_inh=1, this_amp_exc=1,
                                v1_sigma_Ex=0.1, v1_sigma_inh=0.2,
                                v1_amp_inh=1, amp=1,
                                LGN_RF_scale=1, V1_RF_scale=1, v1_amp_exc=1):
    X_vec = np.zeros((timestep, pixel_size + 2 * boundary_size))
    X_vec_2nd = np.zeros((timestep, pixel_size + 2 * boundary_size))
    X_vec_V1 = np.zeros((timestep, pixel_size + 2 * boundary_size))
    X_vec_V1_2nd = np.zeros((timestep, pixel_size + 2 * boundary_size))

    for t in tqdm(range(timestep - 1)):
        # t=0
        this_feedforward, _, _ = fun_feedforward(X_vec[t, :], A, B, C, I[t, :], filter_unit, dt, dt * 2,
                                                 amp_inh=this_amp_inh, RF_scale=LGN_RF_scale, amp_exc=this_amp_exc)
        X_vec[t + 1, :] = X_vec[t, :] + this_feedforward * dt

    for t in tqdm(range(timestep - 1)):
        # t=0
        this_feedforward, _, _ = fun_feedforward(X_vec_V1[t, :], A, B, C, X_vec[t, :], filter_unit_2, v1_sigma_Ex,
                                                 v1_sigma_inh, amp_inh=v1_amp_inh, amp=amp, RF_scale=V1_RF_scale,
                                                 amp_exc=v1_amp_exc)
        X_vec_V1[t + 1, :] = X_vec_V1[t, :] + this_feedforward * dt

    for t in tqdm(range(timestep - 1)):
        this_feedback, _, _ = fun_feedback(X_vec_2nd[t, :], A, B, C, I[t, :], X_vec_V1[t, :], filter_unit, dt, dt * 2,
                                           amp_inh=this_amp_inh, alpha=this_alpha, RF_scale=LGN_RF_scale, amp_exc=this_amp_exc)

        X_vec_2nd[t + 1, :] = X_vec_2nd[t, :] + this_feedback * dt
    for t in tqdm(range(timestep - 1)):
        # t=0
        this_feedforward, _, _ = fun_feedforward(X_vec_V1_2nd[t, :], A, B, C, X_vec_2nd[t, :], filter_unit_2,
                                                 v1_sigma_Ex, v1_sigma_inh, amp_inh=v1_amp_inh, amp=amp,
                                                 RF_scale=V1_RF_scale, amp_exc=v1_amp_exc)
        X_vec_V1_2nd[t + 1, :] = X_vec_V1_2nd[t, :] + this_feedforward * dt
    return X_vec, X_vec_2nd, X_vec_V1, X_vec_V1_2nd


def get_bold_X_vec(dt,  timestep, pixel_size, boundary_size, X_vec, time_res=20):
    response_unit=np.arange(-dt*time_res,dt*(time_res+1),dt)
    G_repsponse= 0.1*stats.norm.pdf(response_unit, loc=0, scale=1)
    X_vec_bold = np.zeros((timestep, pixel_size+2*boundary_size))
    for this_pixel in range(pixel_size):
        X_vec_bold[:,this_pixel]=np.convolve(X_vec[:,this_pixel] , G_repsponse,'same')
    return X_vec_bold


def get_2response_pdf_v(X_vec1, X_vec2):
    this_X_vec1 = X_vec1.copy()
    this_X_vec2 = X_vec2.copy()

    conv = np.convolve(this_X_vec1, np.flip(this_X_vec2), 'same')

    return np.sum(conv[int(len(conv) / 2):])


def get_2response_pdf_cos(X_vec1,X_vec2):
    this_X_vec1 = X_vec1.copy()
    this_X_vec2 = X_vec2.copy()
    return cosine_similarity([this_X_vec1],[this_X_vec2])