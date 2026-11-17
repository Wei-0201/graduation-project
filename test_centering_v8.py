import ast
import unittest
from pathlib import Path
import cv2
import numpy as np
import jetson_centering_v8 as c


def road(shift=0,left=True,right=True):
    f=np.full((240,320,3),190,np.uint8)
    if left:cv2.line(f,(125+shift,72),(35+shift,220),(25,25,25),7)
    if right:cv2.line(f,(195+shift,72),(285+shift,220),(25,25,25),7)
    return f


class V8Tests(unittest.TestCase):
    def detection(self, stamp, offset=0, **fields):
        d=dict(valid=True,quality='dual',offset=offset,timestamp=stamp,target_y=139,width=150)
        d.update(fields)
        return d

    def test_unstable_lane_cannot_authorize_start(self):
        for field,values in (('offset',(-15,15)),('width',(120,150)),('target_y',(130,145))):
            p=c.CommandPolicy()
            for i in range(30):
                now=1+i*.05
                d=self.detection(now,**{field:values[i%2]})
                self.assertEqual(p.command('AUTO','STOP',0,d,now),'STOP')

    def test_sparse_or_old_frames_cannot_authorize_start(self):
        for gap,age in ((.25,0),(.05,.20)):
            p=c.CommandPolicy()
            for i in range(20):
                now=1+i*gap
                self.assertEqual(p.command('AUTO','STOP',0,self.detection(now),now+age),'STOP')

    def test_same_frame_never_counts_as_stable_sequence(self):
        p=c.CommandPolicy()
        for i in range(15):
            self.assertEqual(p.command('AUTO','STOP',0,self.detection(1),1+i*.01),'STOP')
        self.assertLess(p.good_frames,c.ACQUIRE_FRAMES)

    def test_misaligned_heading_prevents_straight_kick(self):
        p=c.CommandPolicy()
        for i in range(20):
            now=1+i*.05
            self.assertEqual(p.command('AUTO','STOP',0,self.detection(now,center_slope=.6),now),'STOP')
        self.assertEqual(p.control_state,'ALIGN HEADING')

    def test_stale_stop_does_not_auto_restart(self):
        p,started=self.start()
        self.assertEqual(p.command('AUTO','STOP',0,None,started+.05),'STOP')
        for i in range(30):
            now=started+.1+i*.05
            self.assertEqual(p.command('AUTO','STOP',0,self.detection(now),now),'STOP')
        self.assertTrue(p.fault)
        p.reset()  # Same reset performed by an explicit AUTO button press.
        for i in range(15):
            now=5+i*.05
            cmd=p.command('AUTO','STOP',0,self.detection(now),now)
        self.assertTrue(cmd.startswith('DRIVE,'))

    def test_heading_divergence_during_straight_start_stops_and_latches(self):
        p,started=self.start()
        now=started+.1
        self.assertEqual(p.command('AUTO','STOP',0,self.detection(now,center_slope=.6),now),'STOP')
        self.assertTrue(p.fault)

    def test_recovery_deadline_cannot_be_extended_by_reacquisition(self):
        p,started=self.start()
        now=started+.75
        p.command('AUTO','STOP',0,self.detection(now),now)
        last=now
        d=self.detection(now+.05,valid=False,quality='none',offset=None)
        self.assertTrue(p.command('AUTO','STOP',0,d,now+.05).startswith('DRIVE,'))
        for delta in (.50,.55):
            now=last+delta
            p.command('AUTO','STOP',0,self.detection(now),now)
        now=last+.61
        self.assertEqual(p.command('AUTO','STOP',0,self.detection(now),now),'STOP')
        self.assertTrue(p.fault)

    def test_fit_matches_least_squares_reference(self):
        rng=np.random.RandomState(5)
        y=np.arange(90,220,4,dtype=float)
        points=np.column_stack((.7*y+20+rng.normal(0,1,len(y)),y)).astype(np.float32)
        np.testing.assert_allclose(c.fit_line(points),np.polyfit(points[:,1],points[:,0],1),rtol=1e-6,atol=1e-6)

    def test_status_includes_version_stop_reason_and_telemetry(self):
        car=c.Car()
        car.receive_esp_line('STATUS PWM_L=154 PWM_R=166')
        car.policy.control_state='STALE CAMERA - PRESS AUTO'
        data=c.make_app(car).test_client().get('/status').get_json()
        self.assertEqual(data['version'],'v8')
        self.assertIn('PRESS AUTO',data['state'])
        self.assertIn('154',data['esp_status'])

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
            d=dict(valid=True,quality='dual',offset=0,timestamp=now,target_y=139)
            command=p.command('AUTO','STOP',0,d,now)
            if command!='STOP':return p,now
        self.fail('Did not start')

    def test_straight_start_then_gradual_lane_correction(self):
        p,started=self.start()
        for dt in (.05,.30,.65):
            now=started+dt
            d=dict(valid=True,quality='dual',offset=0,timestamp=now,target_y=139)
            self.assertEqual(p.command('AUTO','STOP',0,d,now),'DRIVE,0.120,0.0000')
        now=started+.75
        d=dict(valid=True,quality='dual',offset=20,timestamp=now,target_y=139)
        command=p.command('AUTO','STOP',0,d,now)
        turn=float(command.split(',')[2])
        self.assertLess(turn,0)
        self.assertLessEqual(abs(turn),.10)

    def test_straight_start_aborts_on_missing_line(self):
        for quality,offset in (('single',20),('none',None)):
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

    def test_confirm_before_starting_and_correct_large_start_offset(self):
        p=c.CommandPolicy()
        for i in range(8):
            t=1+i*.05;d=dict(valid=True,quality='dual',offset=0,timestamp=t,target_y=139)
            self.assertEqual(p.command('AUTO','STOP',0,d,t),'STOP')
        for i in range(9,14):
            t=1+i*.05;d=dict(valid=True,quality='dual',offset=0,timestamp=t,target_y=139)
            cmd=p.command('AUTO','STOP',0,d,t)
        self.assertTrue(cmd.startswith('DRIVE,0.120,'))
        p=c.CommandPolicy()
        for i in range(20):
            t=1+i*.05;d=dict(valid=True,quality='dual',offset=60,timestamp=t,target_y=139)
            cmd=p.command('AUTO','STOP',0,d,t)
        self.assertTrue(cmd.startswith('DRIVE,0.120,'))
        self.assertLess(float(cmd.split(',')[2]),0)
        self.assertTrue(p.startup_straight_done)

    def test_offset_on_either_side_starts_with_gentle_correct_direction(self):
        for offset in (-60,-40,40,60):
            p=c.CommandPolicy()
            for i in range(20):
                now=1+i*.05
                cmd=p.command('AUTO','STOP',0,self.detection(now,offset),now)
                if cmd!='STOP':break
            self.assertTrue(cmd.startswith('DRIVE,0.120,'))
            turn=float(cmd.split(',')[2])
            self.assertLess(turn*offset,0)
            self.assertLessEqual(abs(turn),.10)
            self.assertTrue(p.startup_straight_done)

    def test_drifting_during_straight_start_engages_correction_instead_of_aborting(self):
        p,started=self.start()
        now=started+.1
        cmd=p.command('AUTO','STOP',0,self.detection(now,45),now)
        self.assertTrue(cmd.startswith('DRIVE,0.120,'))
        self.assertLess(float(cmd.split(',')[2]),0)
        self.assertTrue(p.startup_straight_done)
        self.assertFalse(p.fault)

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
