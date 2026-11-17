import ast
import unittest
from pathlib import Path
import numpy as np
import cv2
from lane_centering_v6 import jetson_centering as c


def detection(t, offset=0, quality='dual'):
    return dict(timestamp=t, offset=offset, quality=quality, valid=quality=='dual', target_y=139)


class CenteringTests(unittest.TestCase):
    def armed(self, offset=-20):
        policy=c.CommandPolicy()
        for t in (1,1.05,1.1):
            command=policy.command('AUTO','STOP',0,detection(t,offset),t)
        self.assertTrue(command.startswith('DRIVE,'))
        return policy

    def test_strength_decreases_as_target_gets_closer(self):
        controller=c.SteeringController()
        # Settled far deviation; then approach center at fixed measured depth.
        for i in range(20): controller.update(45,i*.05)
        strengths=[]
        for i,e in enumerate((45,35,25,15,8,5,0)):
            value=controller.update(e,1+i*.05)
            self.assertLessEqual(value,0)
            strengths.append(abs(value))
        self.assertTrue(all(b<=a for a,b in zip(strengths,strengths[1:])))
        self.assertEqual(strengths[-2:], [0,0])

    def test_sign_symmetry_and_normalized_limits(self):
        for error in (8,20,45,80):
            left,right=c.SteeringController(),c.SteeringController()
            for i in range(30):
                positive=left.update(-error,i*.05)
                negative=right.update(error,i*.05)
                self.assertAlmostEqual(positive,-negative)
                self.assertLessEqual(abs(positive),1)

    def test_center_is_zero_even_with_previous_derivative(self):
        p=c.SteeringController()
        p.update(45,1)
        self.assertEqual(p.update(0,1.05),0)
        self.assertEqual(p.update(-3,1.1),0)

    def test_derivative_brakes_without_reversing_correction(self):
        p=c.SteeringController()
        for i in range(20):p.update(45,i*.05)
        before=abs(p.steer)
        value=p.update(20,1)
        self.assertLessEqual(value,0)
        self.assertLess(abs(value),before)

    def test_no_movement_without_acquisition(self):
        p=c.CommandPolicy()
        for t in (1,1):self.assertEqual(p.command('AUTO','STOP',0,detection(t),t),'STOP')
        self.assertEqual(p.command('AUTO','STOP',0,detection(1.05),1.05),'STOP')
        self.assertTrue(p.command('AUTO','STOP',0,detection(1.1),1.1).startswith('DRIVE,0.120,'))

    def test_stable_ramp_and_unstable_reset(self):
        p=c.CommandPolicy(); speeds=[]
        for i in range(140):
            t=i*.05; cmd=p.command('AUTO','STOP',0,detection(t),t)
            if cmd!='STOP':speeds.append(float(cmd.split(',')[1]))
        self.assertEqual(speeds[0],.12)
        self.assertEqual(speeds[-1],.24)
        self.assertTrue(all(b-a<=c.AUTO_ACCELERATION*.05+.0011 for a,b in zip(speeds,speeds[1:])))
        self.assertEqual(float(p.command('AUTO','STOP',0,detection(7.05,30),7.05).split(',')[1]),.12)

    def test_fresh_memory_expires_without_renewal(self):
        p=self.armed(); last=p.last_dual_time
        for t in (1.15,1.4,1.69):
            cmd=p.command('AUTO','STOP',0,detection(t,None,'none'),t)
            self.assertTrue(cmd.startswith('DRIVE,0.120,'))
            self.assertEqual(p.last_dual_time,last)
        self.assertEqual(p.command('AUTO','STOP',0,detection(1.71,None,'none'),1.71),'STOP')

    def test_camera_failure_stale_ambiguous_and_extreme_offset_stop(self):
        cases=[(None,1.15),(detection(1.15,20,'single'),1.5),
               (detection(1.15,20,'ambiguous'),1.15),(detection(1.15,120,'single'),1.15)]
        for d,t in cases:self.assertEqual(self.armed().command('AUTO','STOP',0,d,t),'STOP')

    def test_never_start_with_single_line(self):
        p=c.CommandPolicy()
        self.assertEqual(p.command('AUTO','STOP',0,detection(1,20,'single'),1),'STOP')

    def test_reacquisition_requires_three_distinct_dual_frames(self):
        p=self.armed();p.command('AUTO','STOP',0,detection(1.15,None,'none'),1.15)
        for t in (1.2,1.2,1.25):
            p.command('AUTO','STOP',0,detection(t),t)
            self.assertTrue(p.recovering)
        cmd=p.command('AUTO','STOP',0,detection(1.3),1.3)
        self.assertFalse(p.recovering)
        self.assertTrue(cmd.startswith('DRIVE,0.120,'))

    def test_old_firmware_blocked_and_new_handshake_required(self):
        car=c.Car();client=c.make_app(car).test_client()
        car.receive_esp_line('READY ESP32 ADAPTIVE TURN 0.95 TO 1.00')
        self.assertFalse(car.connected)
        self.assertEqual(client.post('/mode',json={'mode':'AUTO'}).status_code,409)
        car.receive_esp_line('READY CENTERING V6')
        self.assertTrue(car.connected)
        self.assertEqual(car.mode,'MANUAL')
        self.assertEqual(client.post('/mode',json={'mode':'AUTO'}).status_code,200)
        car.receive_esp_line('EMERGENCY COMMAND TIMEOUT')
        self.assertEqual(car.mode,'MANUAL')
        client.post('/mode',json={'mode':'AUTO'})
        car.receive_esp_line('BOOT CENTERING V6')
        self.assertFalse(car.connected)
        self.assertEqual(car.mode,'MANUAL')
        car.receive_esp_line('READY CENTERING V6')
        self.assertTrue(car.connected)
        self.assertEqual(car.mode,'MANUAL')

    def test_stop_works_in_auto_and_clears_memory(self):
        car=c.Car();car.connected=True;car.mode='AUTO';car.policy=self.armed()
        client=c.make_app(car).test_client()
        self.assertEqual(client.post('/control',json={'command':'STOP'}).status_code,200)
        self.assertIsNone(car.policy.last_dual_time)
        self.assertEqual(car.mode,'MANUAL')

    def test_manual_watchdog_and_late_requests(self):
        p=c.CommandPolicy()
        self.assertEqual(p.command('MANUAL','F',1,None,1.1),'MANUAL,F,0.300,0')
        self.assertEqual(p.command('MANUAL','F',1,None,1.7),'STOP')
        car=c.Car();car.connected=True;client=c.make_app(car).test_client();old=car.generation
        client.post('/control',json={'command':'STOP'})
        self.assertEqual(client.post('/control',json={'command':'F','epoch':old}).status_code,409)

    def test_frame_mailbox_drops_backlog(self):
        box=c.LatestFrame()
        for i in range(10):box.publish(i,100+i/30)
        seq,frame,stamp=box.read_after(0,timeout=0)
        self.assertEqual((seq,frame),(10,9))
        self.assertAlmostEqual(stamp,100+9/30)

    def test_python36_syntax(self):
        ast.parse(Path(c.__file__).read_text(encoding='utf-8'),feature_version=(3,6))


if __name__=='__main__':unittest.main(verbosity=2)
