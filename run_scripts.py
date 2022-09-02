import pandas as pd
import numpy as np
from scipy import optimize

dbm = lambda x: 10 *  np.log10(x)

def check_side(v: float):
    #which side are we grounding? n if V > 0, p if V < 0
    if v > 0:
      apply = 'p'
      gnd = 'n'
    else:
      apply = 'n'
      gnd = 'p'

    return (apply, gnd)

def set_and_read(voltage_df, 
                 devices,
                 device_dict,
                 t_rest: float = 0.05,
                 use_external_detector: bool = True):
    """
    Given a dataframe with one row, set those devices and read their outputs
    """

    #create a dictionary to store the outputs for each device
    output = {}
    for dev in devices:
      output[dev + str("_i")] = np.zeros(len(voltage_df.index))
      output[dev + str("_v")] = np.zeros(len(voltage_df.index))

    if use_external_detector:
      output["external_pd_p"] = np.zeros(len(voltage_df.index))

    #for each device
    for dev in devices:
        #what's the voltage & device we're setting?
        v = voltage_df[dev]
    
        apply, gnd = check_side(v)

        #what pin is it actually mapped to on qontrol
        pin_apply = device_dict[dev][apply]
        pin_gnd = device_dict[dev][gnd]

        #update qontrol values
        q.v[pin_apply] = float(np.abs(v))
        q.v[pin_gnd] = float(0.0)

    #sleep after values are updated
    time.sleep(t_rest)
    #read values
    if use_external_detector:
        output["external_pd_p"] = float(mf.detvisa.query('READ1:POW?'))*1e3

    #for each device
    for dev in devices:
        #check which side is being read for voltage, which side for current
        v = voltage_df[dev]
        apply, gnd = check_side(v)
        read_v = device_dict[dev][apply]
        read_i = device_dict[dev][gnd]

        #read the measured values
        measured_i = q.i[read_i]
        measured_v = q.v[read_v]

        #store them in the output
        output[dev + str("_i")] = measured_i
        output[dev + str("_v")] = measured_v

    return pd.DataFrame(output)

def apply_voltages(voltage_df, 
                   device_dict, 
                   t_rest: float = 0.05, 
                   wl: float = 1520,
                   use_external_detector: bool = True):
    """
    Given a dataframe where each column describes a series of voltages to set for a device on the chip,
    iterate through each row, set the devices, and read the output. Returns a dataframe with currents
    and voltages for each device, as well as the external photodetector.
    """
    # Sweep at fixed wavelength, with voltage sweep (grab ext. detector and on-chip PD)
    mf.tls.wl = wl*1e-9
    mf.visa.write(f"sour0:pow 4.1mW")
    mf.tls.stat.on

    widgets = ['Voltage Sweep:', Bar(), ' ', Percentage(), ' ', ETA()]
    bar = ProgressBar(widgets = widgets, maxval = len(voltage_df.index))
    bar.start()
    k = 0

    devices = list(voltage_df.keys())
    output = pd.DataFrame(columns=devices)

    #for each row of voltages
    for i in voltage_df.index:
        df = voltage_df.iloc[i]
        read = set_and_read(df,
                            devices,
                            device_dict,
                            t_rest=t_rest,
                            use_external_detector=use_external_detector)

        #append to the output
        output = pd.concat([output, read], ignore_index=True)

        #reset the qontrol, ground all pins
        q.v[:] = 0

        #update the progress bar
        bar.update(k)
        k += 1

    mf.tls.stat.off
    bar.finish()

    return output

def apply_voltages_and_seek(voltage_df, 
                            seek_dev,
                            output_dev,
                            device_dict, 
                            t_rest: float = 0.05, 
                            tolerance: float = 0.05,
                            maxiter: int = 50,
                            wl: float = 1520,
                            use_external_detector: bool = True):
    """
    Given a dataframe where each column describes a series of voltages to set for a device on the chip,
    iterate through each row, set the devices, and seek one device in the dataframe for a voltage
    which produces minimum intensity at an output.
    """
    # Sweep at fixed wavelength, with voltage sweep (grab ext. detector and on-chip PD)
    mf.tls.wl = wl*1e-9
    mf.visa.write(f"sour0:pow 4.1mW")
    mf.tls.stat.on

    widgets = ['Voltage Sweep:', Bar(), ' ', Percentage(), ' ', ETA()]
    bar = ProgressBar(widgets = widgets, maxval = len(voltage_df.index))
    bar.start()
    k = 0

    devices = list(voltage_df.keys())
    output = pd.DataFrame(columns=devices)
    niter = []

    #for each row of voltages
    for i in voltage_df.index:
        #get the starting point for the seek
        #make a copy since pd passes by ref
        df = pd.DataFrame(voltage_df.iloc[i])
        #define the sweep function
        def seek_fn(v):
            #change the voltage we're seeking on
            df[seek_dev] = v
            #set & read
            read = set_and_read(df,
                            devices,
                            device_dict,
                            t_rest=t_rest,
                            use_external_detector=use_external_detector)
            #return the output we're looking at
            metric = read[output_dev]
            return metric

        #seek over the sweep function for the voltage which produces minimum intensity
        soln = optimize.minimize_scalar(seek_fn, tol=tolerance, maxiter=maxiter)
        if not soln.success:
            print("Seek failed at: ", df)
        #store results
        niter.append(soln.nit)
        min_v = soln.x
        df[seek_dev] = min_v

        #read again to get the full result
        read = set_and_read(df,
                            devices,
                            device_dict,
                            t_rest=t_rest,
                            use_external_detector=use_external_detector)

        #append to the output
        output = pd.concat([output, read], ignore_index=True)

        #reset the qontrol, ground all pins
        q.v[:] = 0

        #update the progress bar
        bar.update(k)
        k += 1

    mf.tls.stat.off
    bar.finish()

    niter_df = pd.DataFrame({"n_iter" : np.array(niter)})
    output = pd.join([output, niter_df])
    return output