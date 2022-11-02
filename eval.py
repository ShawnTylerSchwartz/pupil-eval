#
# Written by Shawn Schwartz <stschwartz@stanford.edu> 2022
# eval.py
#

from psychopy import visual, core, event, monitors, tools, data, gui, logging 
import sys, os, random, glob, math, csv, uuid, errno, json, pickle, time, pylink, pygaze, platform
import numpy as np
import pandas as pd
from EyeLinkCoreGraphicsPsychoPy.EyeLinkCoreGraphicsPsychoPy.EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
from pygaze import libscreen, libtime, liblog, libinput, eyetracker, _eyetracker, display, keyboard
from pyDOE2 import *

logging.console.setLevel(logging.CRITICAL) # show only critical messages in the PsychoPy console

experiment_name = 'pval'

tracker_types = [
    'tobii',
    'eyelink',
]

env_light_settings = [
    'on',
    'off'
]

def convert_color_value(color):
    """Converts a list of 3 values from 0 to 255 to -1 to 1.
    
    Parameters:
        color -- A list of 3 ints between 0 and 255 to be converted.
    
    Credit: https://github.com/colinquirk/templateexperiments
    """
    
    return [round(((n/127.5)-1), 2) for n in color]

class PupilEval():
    def __init__(self,
                 experiment_name,
                 data_directory,
                 *args, **kwargs):

        super(PupilEval, self).__init__(*args, **kwargs)

        self.experiment_name = experiment_name
        self.data_directory = os.path.join(os.path.expanduser('~'), 'Desktop', self.experiment_name, data_directory)
        self.bg_color = convert_color_value((128, 128, 128))
        self.bg_color_calib = convert_color_value((255, 255, 255)) # for eyelink calibration screen
        self.txt_color = convert_color_value((0, 0, 0))
        self.experiment_window = None
        self.tracker = None
        self.el_tracker = None

        # EyeLink Lab Monitor Settings
        # Mitsubishi diamond pro 2070sb SuperBright Diamondtron CRT display
        self.monitor_name = 'Experiment Monitor'
        self.monitor_width = 40 # cm
        self.monitor_distance = 40 # cm
        self.monitor_px = [2048, 1560]

        self.experiment_monitor = monitors.Monitor(
            self.monitor_name, width=self.monitor_width,
            distance=self.monitor_distance)
        self.experiment_monitor.setSizePix(self.monitor_px)

        vars(self).update(kwargs)

    def get_experiment_info(self):
        self.experiment_info = {
            'Subject id': '',
            'Tracker Type': tracker_types,
            'Room Lighting': env_light_settings,
            'EyeLink IP Address': '100.1.1.1',
        }

        exp_info = gui.DlgFromDict(
            self.experiment_info,
            title = self.experiment_name,
            order = ['Subject id',
                     'Tracker Type',
                     'Room Lighting',
                     'EyeLink IP Address',
            ]
        )

        return exp_info.OK
    
    def start_message(self, tracker):
        msg = 'Now, we will calibrate the eyetracker! \n\nClick the SPACEBAR to begin!'

        if tracker == 'tobii':
            begin_exp_screen = libscreen.Screen(disptype='psychopy')
            begin_exp_text = visual.TextStim(win=pygaze.expdisplay, colorSpace='rgb', color=self.txt_color, text=msg)
            begin_exp_screen.screen.append(begin_exp_text)
            self.experiment_window.fill(screen=begin_exp_screen)
            self.experiment_window.show()
            keys = event.waitKeys(keyList=['space'])
        elif tracker == 'eyelink':
            message = visual.TextStim(win=self.experiment_window, colorSpace='rgb255', color=self.txt_color, text=msg)
            message.draw()
            self.experiment_window.flip()
            keys = event.waitKeys(keyList=['space'])

    def fixation(self, duration, tracker):
        if tracker == 'tobii':
            blank_screen = libscreen.Screen(disptype='psychopy')
            fixation = visual.TextStim(win=pygaze.expdisplay, text='+', height=60, colorSpace='rgb', color=self.txt_color)
            blank_screen.screen.append(fixation)
            self.experiment_window.fill(screen=blank_screen)
            self.experiment_window.show()
        elif tracker == 'eyelink':
            fixation = visual.TextStim(win=self.experiment_window, text='+', height=1.5, colorSpace='rgb255', color=self.txt_color)
            fixation.draw()
            self.experiment_window.flip()

        core.wait(duration)

    def generate_gabor(self, duration, tracker):
        if tracker == 'eyelink':
            #gabor = visual.GratingStim(win=self.experiment_window, tex='sin', mask='gauss', texRes=256, size=[30.0, 30.0], sf=[1, 0], ori=0)
            gabor = visual.GratingStim(win=self.experiment_window, tex='sin', mask='gauss', texRes=256, size=[15.0, 15.0], sf=1/2, ori=0)
            gabor.draw()
            self.experiment_window.flip()
            core.wait(duration)

    def gabor(self, duration, tracker):
        stim_path = 'gabor.png'

        if tracker == 'tobii':
            gabor = visual.ImageStim(win=pygaze.expdisplay, image=stim_path, units='pix', size=[1000, 1000])
            gabor_screen = libscreen.Screen(disptype='psychopy')
            gabor_screen.screen.append(gabor)
            self.experiment_window.fill(screen=gabor_screen)
            self.experiment_window.show()
        elif tracker == 'eyelink':
            gabor = visual.ImageStim(win=self.experiment_window, image=stim_path, units='pix', size=[1000, 1000])
            gabor.draw()
            self.experiment_window.flip()

        core.wait(duration)

    def chdir(self):
        try:
            os.makedirs(self.data_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        os.chdir(self.data_directory)

    def make_subject_dir(self):
        if not os.path.exists(os.path.join(self.experiment_info['Subject id'])):
            os.makedirs(os.path.join(self.experiment_info['Subject id']))

    def setup_eyetracker_and_calibrate(self, calibrate=True, validate=True):
        if not os.path.exists(os.path.join(self.experiment_info['Subject id'], 'eyetracking')):
            os.makedirs(os.path.join(self.experiment_info['Subject id'], 'eyetracking'))

        logfilename = os.path.join(self.experiment_info['Subject id'], 'eyetracking', self.experiment_name + '_sub_' + self.experiment_info['Subject id'] + '_lighting_' + str(self.experiment_info['Room Lighting']) + '_eyetracker')
        logfile = os.path.join(self.data_directory, logfilename)
        tracker = eyetracker.EyeTracker(display=self.experiment_window, trackertype='tobii', logfile=logfile)
        tracker.calibrate(calibrate=calibrate, validate=validate)
        tracker.start_recording()
        tracker.status_msg("beginning_of_experiment")
        tracker.log("sub_%s_start_%s_experiment" % (self.experiment_info['Subject id'], self.experiment_name))

        return tracker

    def setup_eyetracker(self, calibrate=True, validate=True):
        self.tracker = self.setup_eyetracker_and_calibrate(calibrate, validate)
        core.wait(0.05)

    def build_trials(self, tracker, n_reps=24):
        fix_time = .25 #5 #0.25
        gabor_time = .25 #5 #0.5

        for i in range(0, n_reps):
            if tracker == 'tobii':
                self.tracker.log("fixation_%d" % (i))
                self.fixation(fix_time, 'tobii')
                self.tracker.log("gabor_%d" % (i))
                self.gabor(gabor_time, 'tobii')
            elif tracker == 'eyelink':
                self.el_tracker.sendMessage('fixation_%d' % (i))
                self.fixation(fix_time, 'eyelink')
                self.el_tracker.sendMessage('gabor_%d' % (i))
                self.gabor(gabor_time, 'eyelink')

    def quit_experiment(self):
        self.experiment_window.close()

    def run(self):
        self.chdir()

        ok = self.get_experiment_info()

        if not ok:
            print("Experiment has ended.")
            sys.exit(1)

        if self.experiment_window is None:
            if self.experiment_info['Tracker Type'] == 'tobii':
                self.experiment_window = libscreen.Display(disptype='psychopy')
            elif self.experiment_info['Tracker Type'] == 'eyelink':
                self.experiment_window = visual.Window(
                    monitor=self.experiment_monitor, fullscr=True, color=self.bg_color, winType='pyglet',
                    colorSpace='rgb', units='deg', allowGUI=False)
                self.experiment_window.mouseVisible = False

        self.make_subject_dir()

        # self.generate_gabor(10, 'eyelink')

        if self.experiment_info['Tracker Type'] == 'tobii':
            self.setup_eyetracker()
            self.start_message('tobii')
            self.build_trials('tobii')
            self.tracker.stop_recording()
        elif self.experiment_info['Tracker Type'] == 'eyelink':
            self.experiment_window.color = self.bg_color_calib
            self.experiment_window.flip()
            try:
                self.el_tracker = pylink.EyeLink(self.experiment_info['EyeLink IP Address'])
            except RuntimeError as error:
                print('ERROR:', error)
                self.experiment_window.close()
                sys.exit(1)

            self.edf_file = str(self.experiment_info['Subject id'])[0:3] + str(self.experiment_info['Room Lighting']) + '.EDF'

            try:
                self.el_tracker.openDataFile(self.edf_file)
            except RuntimeError as err:
                print('ERROR:', err)
                # close the link if one is open
                if self.el_tracker.isConnected():
                    self.el_tracker.close()
                self.experiment_window.close()
                sys.exit(1)

            # third, add preamble text
            preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
            self.el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)

            # fourth, configure the tracker
            self.el_tracker.setOfflineMode() # put the tracker in offline mode before tracking parameters are changed
            eyelink_ver = 0
            vstr = self.el_tracker.getTrackerVersionString()
            eyelink_ver = int(vstr.split()[-1].split('.')[0])
            print('Running experiment on %s, version %d' % (vstr, eyelink_ver))

            file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
            link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'

            if eyelink_ver > 3:
                file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
                link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
            else:
                file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
                link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
            self.el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
            self.el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
            self.el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
            self.el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

            if eyelink_ver > 2:
                self.el_tracker.sendCommand("sample_rate 1000")

            self.el_tracker.sendCommand("calibration_type = HV9")
            self.el_tracker.sendCommand("randomize_calibration_order = NO")
            self.el_tracker.sendCommand("calibration_area_proportion 1.0 1.0")
            self.el_tracker.sendCommand("validation_area_proportion 0.6 0.6")

            self.el_tracker.sendCommand("button_function 5 'accept_target_fixation'")

            # get the native screen resolution used by psychopy
            scn_width = int(self.experiment_window.size[0])
            scn_height = int(self.experiment_window.size[1])

            # pass the display pixel coordinates (left, top, right, bottom) to the tracker
            # el_coords = "screen_pixel_coords = 0.0, 0.0, 1920.0, 1080.0" # hard coded version (use this if the line below doesn't work or gives weird retina behavior on mac)
            el_coords = "screen_pixel_coords = 0.0, 0.0, %d, %d" % (scn_width, scn_height)
            self.el_tracker.sendCommand(el_coords)

            # write a DISPLAY_COORDS message to the EDF file
            dv_coords = "DISPLAY_COORDS 0 0 %d %d" % (scn_width, scn_height)
            self.el_tracker.sendMessage(dv_coords)

            calib_x0 = 400
            calib_x1 = 960
            calib_x2 = 1420
            calib_y0 = 300
            calib_y1 = 540
            calib_y2 = 780
            
            calib_positions = "calibration_targets = %d,%d %d,%d %d,%d %d,%d %d,%d %d,%d %d,%d %d,%d %d,%d" % (
                calib_x1, calib_y1, 
                calib_x1, calib_y0, 
                calib_x1, calib_y2, 
                calib_x0, calib_y1, 
                calib_x2, calib_y1, 
                calib_x0, calib_y0, 
                calib_x2, calib_y0, 
                calib_x0, calib_y2,
                calib_x2, calib_y2)
            self.el_tracker.sendCommand(calib_positions)

            self.start_message('eyelink')

            genv = EyeLinkCoreGraphicsPsychoPy(self.el_tracker, self.experiment_window)

            pylink.openGraphicsEx(genv)

            self.el_tracker.doTrackerSetup()
            
            self.el_tracker.startRecording(1, 1, 1, 1)
            time.sleep(.1)  # required

            self.el_tracker.sendMessage('start_run')

            self.experiment_window.color = self.bg_color
            self.experiment_window.flip()

            self.build_trials('eyelink')

            self.el_tracker.sendMessage('end_run')

            if self.el_tracker.isConnected():
                time.sleep(.1)  # required
                self.el_tracker.stopRecording()
                
                # put the tracker into offline mode
                self.el_tracker.setOfflineMode()

                # clear the host pc screen and wait for 500 ms
                self.el_tracker.sendCommand('clear_screen 0')
                pylink.msecDelay(500)

                # close the EDF data file on the host pc
                self.el_tracker.closeDataFile()

                # show a file transfer message on the screen
                # message = visual.TextStim(win=self.experiment_window, colorSpace='rgb255', color=self.text_color, text="EDF data is transferring from EyeLink Host PC...")
                message = visual.TextStim(win=self.experiment_window, colorSpace='rgb255', color=self.text_color, text="Disconnecting EyeLink...")
                message.draw()
                self.experiment_window.flip()

                # disconnect the tracker
                self.el_tracker.close()

        self.quit_experiment()

if __name__ == '__main__':
    exp = PupilEval(
        experiment_name = experiment_name,
        data_directory = 'Data'
    )

    exp.run()