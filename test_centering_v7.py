import ast
import unittest
from pathlib import Path
import cv2
import numpy as np
import jetson_centering_v7 as c


def road(shift=0,left=True,right=True):
    f=np.full((240,320,3),190,np.uint8)
    if left:cv2.line(f,(125+shift,72),(35+shift,220),(25,25,25),7)
    if right:cv2.line(f,(195+shift,72),(285+shift,220),(25,25,25),7)
    return f


class V7Tests(unittest.TestCase):
    def test_hysteresis_ignores_small_offsets_until_outer_trigger(self):
        p=c.SteeringController()
        for i,e in enumerate((0,8,12,15,19,-8,-15,-19)):
            self.assertEqual(p.update(e,1+i*.05),0)
        self.assertLess(p.update(20,1.5),0)
        self.assertTrue(p.correcting)

    def test_correction_stops_at_inner_band_and_does_not_restart_immediately(self):
        p=c.SteeringController()
        for i in range(10):p.update(30,i*.05)
        p.update(16,.55)
        self.assertTrue(p.correcting)
        self.assertEqual(p.update(12,.60),0)
        self.assertFalse(p.correcting)
        for i,e in enumerate((13,15,18,19)):
            self.assertEqual(p.update(e,.65+i*.05),0)
        self.assertLess(p.update(21,.9),0)

    def test_crossing_center_does_not_immediately_correct_opposite_side(self):
        p=c.SteeringController()
        for i in range(10):p.update(-30,i*.05)
        self.assertEqual(p.update(15,.55),0)
        self.assertFalse(p.correcting)
        self.assertLess(p.update(25,.6),0)

    def start(self):
        p=c.CommandPolicy()
        for i in range(20):
            now=1+i*.05
            d=dict(valid=True,quality='dual',offset=20,timestamp=now,target_y=139)
            command=p.command('AUTO','STOP',0,d,now)
            if command!='STOP':return p,now
        self.fail('Did not start')

    def test_straight_start_then_gradual_lane_correction(self):
        p,started=self.start()
        for dt in (.05,.30,.65):
            now=started+dt
            d=dict(valid=True,quality='dual',offset=20,timestamp=now,target_y=139)
            self.assertEqual(p.command('AUTO','STOP',0,d,now),'DRIVE,0.120,0.0000')
        now=started+.75
        d=dict(valid=True,quality='dual',offset=20,timestamp=now,target_y=139)
        command=p.command('AUTO','STOP',0,d,now)
        turn=float(command.split(',')[2])
        self.assertLess(turn,0)
        self.assertLessEqual(abs(turn),.10)

    def test_straight_start_aborts_on_missing_line_or_large_offset(self):
        for quality,offset in (('single',20),('none',None),('dual',60)):
            p,started=self.start();now=started+.10
            d=dict(valid=quality=='dual',quality=quality,offset=offset,timestamp=now,target_y=139)
            self.assertEqual(p.command('AUTO','STOP',0,d,now),'STOP')

    def test_memory_reacquisition_does_not_force_another_straight_phase(self):
        p,started=self.start()
        for dt in (.30,.65,.75,1.0):
            now=started+dt
            d=dict(valid=True,quality='dual',offset=20,timestamp=now,target_y=139)
            p.command('AUTO','STOP',0,d,now)
        self.assertTrue(p.startup_straight_done)
        now=started+1.05
        d=dict(valid=False,quality='single',offset=20,timestamp=now,target_y=139)
        p.command('AUTO','STOP',0,d,now)
        for dt in (1.10,1.15,1.20):
            now=started+dt
            d=dict(valid=True,quality='dual',offset=20,timestamp=now,target_y=139)
            p.command('AUTO','STOP',0,d,now)
        self.assertTrue(p.startup_straight_done)
        self.assertNotEqual(p.control_state,'START STRAIGHT')

    def test_visible_floor_lanes_and_offsets(self):
        for shift in (-25,0,25):
            r=c.LaneDetector().process(road(shift),1)[2]
            self.assertTrue(r['valid'])
            self.assertAlmostEqual(r['offset'],shift,delta=3)

    def test_background_only_cannot_become_a_lane(self):
        f=np.full((240,320,3),190,np.uint8)
        cv2.line(f,(95,55),(30,110),(25,25,25),7)
        cv2.line(f,(190,55),(285,110),(25,25,25),7)
        self.assertFalse(c.LaneDetector().process(f,1)[2]['valid'])

    def test_vertical_furniture_without_tape_is_rejected(self):
        f=np.full((240,320,3),190,np.uint8)
        for x in (35,90,220,280):cv2.line(f,(x,75),(x+4,219),(25,25,25),5)
        self.assertFalse(c.LaneDetector().process(f,1)[2]['valid'])

    def test_background_with_real_floor_does_not_hide_the_lane(self):
        f=road()
        for x in range(10,320,19):cv2.line(f,(x,30),(x,105),(15,15,15),3)
        r=c.LaneDetector().process(f,1)[2]
        self.assertTrue(r['valid'])
        self.assertLess(abs(r['offset']),3)

    def test_confirm_before_starting_and_refuse_large_start_offset(self):
        p=c.CommandPolicy()
        for i in range(9):
            t=1+i*.05;d=dict(valid=True,quality='dual',offset=0,timestamp=t,target_y=139)
            self.assertEqual(p.command('AUTO','STOP',0,d,t),'STOP')
        for i in range(9,14):
            t=1+i*.05;d=dict(valid=True,quality='dual',offset=0,timestamp=t,target_y=139)
            cmd=p.command('AUTO','STOP',0,d,t)
        self.assertTrue(cmd.startswith('DRIVE,0.120,'))
        p=c.CommandPolicy()
        for i in range(20):
            t=1+i*.05;d=dict(valid=True,quality='dual',offset=60,timestamp=t,target_y=139)
            self.assertEqual(p.command('AUTO','STOP',0,d,t),'STOP')

    def test_missing_camera_stops_and_forgets_recovery(self):
        p=c.CommandPolicy()
        for i in range(14):
            t=1+i*.05;d=dict(valid=True,quality='dual',offset=0,timestamp=t,target_y=139)
            cmd=p.command('AUTO','STOP',0,d,t)
        self.assertTrue(cmd.startswith('DRIVE,'))
        self.assertEqual(p.command('AUTO','STOP',0,None,t+.05),'STOP')
        d=dict(valid=False,quality='single',offset=0,timestamp=t+.1,target_y=139)
        self.assertEqual(p.command('AUTO','STOP',0,d,t+.1),'STOP')

    def test_single_side_recovery_has_expiry(self):
        d=c.LaneDetector()
        for t in (1,1.05,1.1):d.process(road(),t)
        r=d.process(road(left=False),1.15)[2]
        self.assertEqual(r['quality'],'single')
        self.assertFalse(r['valid'])
        self.assertEqual(d.last_good,1.1)
        self.assertEqual(d.process(road(left=False),1.71)[2]['quality'],'none')

    def test_python36_syntax(self):
        ast.parse(Path(c.__file__).read_text(encoding='utf-8'),feature_version=(3,6))


if __name__=='__main__':unittest.main(verbosity=2)
