"""Vectorized task math matching the released CAT and whole-body extensions.

These functions never advance physics or sample randomness. Numerical parity
fixtures can therefore separate task semantics from simulator differences.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def quat_conj(q):
    return torch.cat((q[...,:1],-q[...,1:]),-1)


def quat_mul(a,b):
    aw,ax,ay,az=a.unbind(-1); bw,bx,by,bz=b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw),-1)


def quat_rotate(q,v):
    # Expanded multiply exactly matches the legacy unit-quaternion helper.
    return quat_mul(quat_mul(q,torch.cat((torch.zeros_like(v[...,:1]),v),-1)),quat_conj(q))[...,1:]


def delay_body_pos(qpos,odom,positions):
    local=quat_rotate(quat_conj(qpos[:,None,3:7]),positions-qpos[:,None,:3])
    return odom[:,None,:3]+quat_rotate(odom[:,None,3:7],local)


def navi_rotation(rotation):
    x=rotation[:,:,0].clone(); x[:,2]=0
    x=x/torch.linalg.vector_norm(x,dim=-1,keepdim=True)
    z=torch.zeros_like(x); z[:,2]=1
    y=torch.linalg.cross(z,x,dim=-1)
    y=y/torch.linalg.vector_norm(y,dim=-1,keepdim=True)
    return torch.stack((torch.linalg.cross(y,z,dim=-1),y,z),-1)


def world_to_navi(rotation,values):
    return torch.einsum('bij,b...i->b...j',rotation,values)


def matrix_rpy(matrix):
    # Matches SO3.as_rpy_radians away from its Euler singularity.
    pitch=torch.atan2(-matrix[:,2,0],torch.sqrt(matrix[:,0,0].square()+matrix[:,1,0].square()))
    return torch.stack((torch.atan2(matrix[:,2,1],matrix[:,2,2]),pitch,
                        torch.atan2(matrix[:,1,0],matrix[:,0,0])),-1)


def motor_targets(action,previous,nominal,lower,upper,*,action_scale=.5,upper_action_scale=.8,
                  upper_target_rate=2.,dt=.02):
    legs=torch.maximum(torch.minimum(previous[:,:12]+action[:,:12]*action_scale,upper[:12]),lower[:12])
    requested=nominal[12:]+action[:,12:]*upper_action_scale
    delta=upper_target_rate*dt
    arms=torch.maximum(torch.minimum(requested,previous[:,12:]+delta),previous[:,12:]-delta)
    arms=torch.maximum(torch.minimum(arms,upper[12:]),lower[12:])
    return torch.cat((legs,arms),-1)


def pd_torque(joint_pos,joint_vel,targets,kps,kds,kp_scale,kd_scale,rfi_scale,noise,torque_limit,feedforward=None):
    torque=(kp_scale[:,None]*kps)*(targets-joint_pos)+(kd_scale[:,None]*kds)*(-joint_vel)
    if feedforward is not None:torque=torque+feedforward
    torque=torque+rfi_scale*noise
    return torch.maximum(torch.minimum(torque,torque_limit),-torque_limit)


def compute_cmd_from_rtf(rtf,cgf,cbf):
    v=rtf[:,:2]*.7
    bhat=cbf[:,:,:2]/(torch.linalg.vector_norm(cbf[:,:,:2],dim=-1,keepdim=True)+1e-9)
    limits=(bhat*cgf[:,:,:2]).sum(-1)
    projected=(bhat*v[:,None]).sum(-1)
    diff=(limits-projected)/(bhat.square().sum(-1)+1e-9)
    delta=torch.where((limits>projected)[:,:,None],diff[:,:,None]*bhat,0.)
    velocity=v+delta.mean(1)
    command=torch.cat((torch.ones_like(v[:,:1]),velocity,torch.zeros_like(v[:,:1])),-1)*.75
    return torch.where((torch.linalg.vector_norm(command[:,1:],dim=-1)<.2)[:,None],0.,command)


def update_phase(command,last_command,stop_timestep,phase,phase_dt,gait_bound,room_enabled):
    stop=torch.where(room_enabled&(command[:,0]>.5),100,stop_timestep)
    before=stop>50; during=~before&(stop>0); after=~before&~during
    move2stop=(last_command[:,0]==1.)&(command[:,0]==0.)&before
    stop=torch.where(move2stop,50,stop)
    stop=torch.where(during,stop-1,stop)
    command=torch.where(before[:,None],command,0.).clone()
    command[:,0]=torch.where(after,0.,1.)
    phase=torch.fmod(phase+phase_dt[:,None]+torch.pi,2*torch.pi)-torch.pi
    phase=torch.where(after[:,None],0.,phase)
    cosine=phase.cos()
    gait=torch.where(cosine>gait_bound,1.,torch.where(cosine < -gait_bound,-1.,0.))
    return command,stop,phase,gait


def pack_fields(gf,bf,sdf):
    chunks=[]
    for a,b in ((0,1),(1,2),(2,3),(3,5),(5,7),(7,9),(9,11)):
        chunks.extend((gf[:,a:b].flatten(1),bf[:,a:b].flatten(1),sdf[:,a:b].flatten(1)))
    return torch.cat(chunks,-1)


def observations(*,joint_pos,joint_vel,nominal,gyro,gravity,linear_velocity,noise,last_action,
                 targets,command,command_delay,foot_height,phase,navi,gf,bf,sdf,
                 gf_delay,bf_delay,sdf_delay,positions,velocities,torso_rpy,gait,contacts,
                 kp,kd,rfi,elbow_true,elbow_actor):
    gait_phase=torch.cat((phase.cos(),phase.sin()),-1)
    common=lambda gy,gr,q,v,cmd: torch.cat((gy,gr,q-nominal,v,last_action,targets,cmd,
                                          foot_height[:,None],gait_phase),-1)
    true=common(gyro,gravity,joint_pos,joint_vel,command)
    delayed_command=command.clone()
    delayed_command[:,1:]=world_to_navi(navi,command_delay[:,1:])
    delayed_command[:,-1]=0.
    noisy=common(gyro+noise['gyro'],gravity+noise['gravity'],joint_pos+noise['joint_pos'],
                 joint_vel+noise['joint_vel'],delayed_command)
    actor_fields=pack_fields(world_to_navi(navi,gf_delay),
                            world_to_navi(navi,bf_delay)*(sdf_delay<.5),sdf_delay.clamp(-1.,.5))
    hints=torch.cat((positions[:,0],velocities[:,0],positions[:,1],positions[:,2],
                     positions[:,3:5].flatten(1),velocities[:,3:5].flatten(1),
                     positions[:,5:7].flatten(1),velocities[:,5:7].flatten(1),
                     positions[:,7:9].flatten(1),positions[:,9:11].flatten(1),
                     torso_rpy[:,:2],gait,contacts.float(),kp[:,None],kd[:,None],rfi),-1)
    return {'state':torch.nan_to_num(torch.cat((noisy,actor_fields,elbow_actor),-1)),
            'privileged_state':torch.nan_to_num(torch.cat((true,linear_velocity,
                pack_fields(gf,bf,sdf),hints,elbow_true),-1))}


def hand_clearance_pressure(distance,*,target=.04,anticipation=.20,near_weight=.8):
    distance=distance.reshape(-1,2)
    near=((target-distance)/target).clamp(0,1)
    early=((anticipation-distance)/anticipation).clamp(0,1)
    pressure=near_weight*near.square()+(1-near_weight)*early.square()
    return pressure.mean(-1),dict(hand_clearance_pressure=pressure.mean(-1),
        left_hand_clearance_pressure=pressure[:,0],right_hand_clearance_pressure=pressure[:,1],
        hand_protection_active_fraction=(distance<anticipation).float().mean(-1))


def upper_stability_terms(target,previous,previous_previous,nominal,handsdf,elbow_clearance,*,
        dt=.02,velocity_scale=2.,acceleration_scale=20.,posture_scale=.8,hand_margin=.12,
        elbow_margin=.08,clearance_taper=.12,waist_weight=4.,hand_protection=False,
        arm_velocity_weight=.5,arm_acceleration_weight=.1,protection_target_clearance=.04,
        protection_anticipation_distance=.20,protection_near_weight=.8,protection_posture_taper=.10,
        contrast_arm_active=None):
    weights=torch.ones_like(nominal);weights[:3]=waist_weight
    mean=lambda x:(weights*x).sum(-1)/weights.sum()
    velocity=(target-previous)/dt; acceleration=(target-2*previous+previous_previous)/(dt*dt)
    offset=target-nominal
    comfortable=torch.minimum(((handsdf.reshape(-1,2)-hand_margin)/clearance_taper).amin(-1),
                               ((elbow_clearance-elbow_margin)/clearance_taper).amin(-1))
    blend=comfortable.clamp(0,1); gate=blend.square()*(3-2*blend)
    costs={'wholebody_upper_target_velocity':mean((velocity/velocity_scale).square()),
           'wholebody_upper_target_acceleration':mean((acceleration/acceleration_scale).square()),
           'wholebody_upper_clear_posture':gate*mean((offset/posture_scale).square())}
    if hand_protection:
        comfortable=torch.minimum((handsdf.reshape(-1,2)-protection_anticipation_distance)/protection_posture_taper,
                                   (elbow_clearance-elbow_margin)/protection_posture_taper)
        blend=comfortable.clamp(0,1);arm_gates=blend.square()*(3-2*blend)
        if contrast_arm_active is not None:arm_gates=arm_gates*~contrast_arm_active
        gates=torch.cat((torch.ones_like(target[:,:3]),arm_gates.repeat_interleave(7,-1)),-1)
        vw=weights.clone();vw[3:]*=arm_velocity_weight
        aw=weights.clone();aw[3:]*=arm_acceleration_weight
        costs={'wholebody_upper_target_velocity':(vw*(velocity/velocity_scale).square()).sum(-1)/weights.sum(),
               'wholebody_upper_target_acceleration':(aw*(acceleration/acceleration_scale).square()).sum(-1)/weights.sum(),
               'wholebody_upper_clear_posture':mean(gates*(offset/posture_scale).square())}
        gate=arm_gates.mean(-1)
    elif contrast_arm_active is not None:
        gates=torch.cat((gate[:,None].expand(-1,3),(gate[:,None]*~contrast_arm_active).repeat_interleave(7,-1)),-1)
        costs['wholebody_upper_clear_posture']=mean(gates*(offset/posture_scale).square())
    rms=lambda x:x.square().mean(-1).sqrt()
    telemetry=dict(upper_target_velocity_rms=rms(velocity),upper_target_acceleration_rms=rms(acceleration),
        upper_target_offset_rms=rms(offset),upper_target_distance_normalized=rms(offset/posture_scale),
        waist_target_offset_rms=rms(offset[:,:3]),clear_posture_gate=gate,
        minimum_hand_clearance=handsdf.reshape(-1,2).amin(-1),minimum_elbow_clearance=elbow_clearance.amin(-1),
        hand_penetration=(handsdf.reshape(-1,2)<0).any(-1).float(),elbow_penetration=(elbow_clearance<0).any(-1).float())
    if hand_protection:
        _,pressure=hand_clearance_pressure(handsdf,target=protection_target_clearance,
            anticipation=protection_anticipation_distance,near_weight=protection_near_weight)
        telemetry.update(pressure)
        telemetry.update(left_arm_posture_gate=arm_gates[:,0],right_arm_posture_gate=arm_gates[:,1],
            arm_target_offset_rms=rms(offset[:,3:]),left_arm_target_offset_rms=rms(offset[:,3:10]),
            right_arm_target_offset_rms=rms(offset[:,10:]),arm_target_velocity_rms=rms(velocity[:,3:]),
            arm_target_acceleration_rms=rms(acceleration[:,3:]),left_shoulder_pitch_target_offset=offset[:,3],
            right_shoulder_pitch_target_offset=offset[:,10],left_elbow_target_offset=offset[:,6],right_elbow_target_offset=offset[:,13])
    return costs,telemetry


def gf_reward(guidance,velocity,sdf,crossed,*,tau,standing=None,standing_value=4.):
    """Guidance-velocity alignment, with two constant cases kept distinct.

    ``crossed`` is a genuine goal/plane event and keeps its 4.0 bonus. ``standing`` was
    previously folded into the same constant, which meant any step with no movement
    command earned a fixed 4.0 per group that no action could change. With headgf 1 +
    handsgf 1 + feetgf 2 that is +16.0/step, and it was measured at ~54% of the mean
    per-step reward once standing scenes reached half the batch: half the compute
    produced no policy gradient. ``standing_value`` makes that price explicit.
    """
    guidance=guidance/(torch.linalg.vector_norm(guidance,dim=-1,keepdim=True)+1e-6)
    velocity=velocity/(torch.linalg.vector_norm(velocity,dim=-1,keepdim=True)+1e-6)
    near=torch.sigmoid(40.*(tau-sdf.squeeze(-1)))*5.*(guidance*velocity).sum(-1)
    if standing is not None:
        near=torch.where(standing,torch.as_tensor(standing_value,dtype=near.dtype,device=near.device),near)
    return torch.where(crossed,4.,near).mean(-1)


def sdf_reward(sdf, knee=None):
    legacy = (-20.*F.softplus((.05-sdf)/.02)).flatten(1).mean(-1)
    if knee is None:
        return legacy
    modified = (-20.*F.softplus((knee[:, None, None]-sdf)/.02)).flatten(1).mean(-1)
    # Preserve the exact legacy arithmetic for every unmodified/outside row.
    return torch.where(knee == .05, legacy, modified)


def posture_terms(torso_pitch,head_z,head_guidance,head_sdf,torso_angvel,*,head_target=1.20,head_scale=.08,
                  pitch_scale=torch.pi/18,crouch_field_z=-.3,crouch_sdf=.20):
    """Straight back and full height, unless the head field asks for a crouch.

    tracking_orientation ignores torso pitch whenever the head is below 1.1 m, so a policy
    can drop its head just under that line and hunch for free (measured on 17 walks:
    27-52 deg mean torso pitch, pelvis 10-14 cm low, head 0.92-1.13 m). These terms pay
    for standing tall and upright and switch off only for a REASON: the head guidance
    field points down (a hurdle/crouch module) or something is within ``crouch_sdf`` of
    the head. CAT's crouch scenes keep working because their head field dips exactly there.
    upright and stand_tall are bonuses in [0,1]; torso_rate is a cost that damps bobbing.
    """
    crouch=(head_guidance[:,2]<crouch_field_z)|(head_sdf<crouch_sdf)
    free=(~crouch).float()
    upright=torch.exp(-(torso_pitch/pitch_scale).square())*free
    stand_tall=torch.exp(-((head_target-head_z).clamp_min(0)/head_scale).square())*free
    torso_rate=torso_angvel[:,:2].square().sum(-1)
    return dict(upright=upright,stand_tall=stand_tall,torso_rate=torso_rate),crouch


def segment_distance(points,a,b):
    """Distance from points [N,H,3] to segments a->b [N,K,3]; returns [N,H,K]."""
    ab=b-a;ap=points[:,:,None]-a[:,None]
    t=((ap*ab[:,None]).sum(-1)/(ab*ab).sum(-1).clamp_min(1e-9)[:,None]).clamp(0.,1.)
    closest=a[:,None]+t[...,None]*ab[:,None]
    return torch.linalg.vector_norm(points[:,:,None]-closest,dim=-1)


def self_clearance_terms(hands,hand_radii,segment_a,segment_b,segment_radii,*,margin=.04):
    """Hand envelope spheres against the robot's own leg capsules (thigh, shin per side).

    Surface gap = centre distance - hand radius - capsule radius. Per hand/capsule pair the
    cost is a quadratic ramp from 0 at ``margin`` of clearance to 1 at contact, held at 1
    inside (the contact pairs stop penetration physically; the reward does not need to
    grow without bound and drag the sum through the floor). Measured before this term
    existed: fingers within 2 cm of a leg in 40-64% of walking frames and penetrating in
    8 of 17 walks. Returns (cost [N], min gap [N]).
    """
    gap=segment_distance(hands,segment_a,segment_b)-hand_radii.reshape(1,-1,1)-segment_radii.reshape(1,1,-1)
    cost=((margin-gap).clamp(0.,margin)/margin).square().sum((1,2))
    return cost,gap.amin((1,2))


def heading_probe_points(root_xy,direction,height,*,half_width=.16,lookahead=.30):
    """Where the shoulders WOULD be if the body faced ``direction``: here and one stride ahead.

    root_xy [N,2] pelvis xy, direction [N,2] unit command direction (zero when idle), height [N]
    probe height (shoulder z). Returns [N,4,3] world points. The G1 shoulder pitch joints sit at
    y=+-0.100 m in the MJCF and the upper-arm capsule adds ~0.06 m, hence 0.16.
    """
    normal=torch.stack((-direction[:,1],direction[:,0]),-1)
    ahead=lookahead*direction
    offsets=torch.stack((half_width*normal,-half_width*normal,ahead+half_width*normal,ahead-half_width*normal),1)
    xy=root_xy[:,None]+offsets
    return torch.cat((xy,height[:,None,None].expand(-1,offsets.shape[1],1)),-1)


def heading_align_reward(direction,pelvis_yaw,move,probe_sdf,*,margin_low=.05,margin_high=.15):
    """Bonus for facing the guidance direction, switched off where a forward-facing body would not fit.

    align is 1 facing forward, .5 sideways, 0 backwards. gate ramps from 0 at ``margin_low`` of
    shoulder clearance to 1 at ``margin_high`` using the WORST probe, so it is closed inside and
    just before any gap too narrow to face forward through. Bonus form in [0,1]: it never pushes
    the reward sum toward the floor, and sidling where the gate is closed costs nothing -- it
    simply stops earning the forward bonus. This replaces the retired contrast heading cost,
    which only ever fired inside authored contrastive passages.
    """
    facing=torch.stack((pelvis_yaw.cos(),pelvis_yaw.sin()),-1)
    commanded=(direction.abs().sum(-1)>0).float()
    align=.5*(1.+(facing*direction).sum(-1))*commanded
    gate=((probe_sdf.amin(-1)-margin_low)/(margin_high-margin_low)).clamp(0.,1.)
    return align*gate*move


def native_rewards(*,action,last_action,last_last_action,joint_pos,joint_vel,last_joint_vel,
        lower,upper,actuator_force,command,pelvis_rpy,torso_rpy,head_z,torso_height,
        global_velocity,torso_angvel,navi,leg_rotations,feet_pos,feet_sensor_velocity,
        subtree_com,feet_contact,gait,foot_height,foot_height_stance,gf,positions,velocities,sdf,
        crossed,dt=.02,max_yaw=.5,sdf_knee=None,standing_gf=4.,heading_sdf=None,heading_margins=(.05,.15)):
    move=command[:,0];cmd=command[:,1:]
    pitch_negative=torso_rpy[:,1].clamp(-torch.pi,0).abs()
    orientation=pelvis_rpy[:,0].abs()+torso_rpy[:,0].abs()+pitch_negative+(head_z>torso_height+.1)*torso_rpy[:,1].abs()
    cmd_norm=torch.linalg.vector_norm(cmd[:,:2],dim=-1)
    zero=torch.isclose(cmd_norm,torch.zeros_like(cmd_norm))
    direction=torch.where(zero[:,None],0.,cmd[:,:2]/cmd_norm.clamp_min(1e-30)[:,None])
    orth=global_velocity[:,:2]-(global_velocity[:,:2]*direction).sum(-1)[:,None]*direction
    motion=1.2*torch.where(zero,0.,orth.square().sum(-1))+.4*torso_angvel[:,:2].abs().sum(-1)
    legs=torch.einsum('bji,bsjk->bsik',navi,leg_rotations)
    decay=((max_yaw+1e-6-cmd[:,2].abs())/(max_yaw+1e-6)).clamp(0,1).square()
    axis=legs[:,:,2,1].abs().mean(-1)+decay*legs[:,:,0,1].abs().mean(-1)
    stance=feet_contact.float();swing=1-stance
    contact=((stance-(gait==1).float()).abs()+(swing-(gait==-1).float()).abs())*(gait!=0)
    foot_distance=torch.linalg.vector_norm(feet_pos[:,0]-feet_pos[:,1],dim=-1)
    spread=(.35-foot_distance).clamp_min(0)
    feet_local=world_to_navi(navi,feet_pos-subtree_com[:,None])
    balance=feet_local[:,:,:2].sum(1).square().sum(-1)*(1+10*spread)
    smooth=action.square()+(action-last_action).square()+(action-2*last_action+last_last_action).square()
    limits=(lower-joint_pos).clamp_min(0)+(joint_pos-upper).clamp_min(0)
    rewards=dict(tracking_orientation=torch.exp(-.5*orientation)-pitch_negative,
        tracking_root_field=torch.exp(-4*(cmd[:,:2]-global_velocity[:,:2]).square().sum(-1)),
        body_motion=motion,body_rotation=torch.exp(-5*axis),foot_contact=contact.sum(-1)*move,
        foot_clearance=(((foot_height_stance+foot_height[:,None]-feet_pos[:,:,2]).clamp_min(0)).square()*(gait==-1)).sum(-1)*move,
        foot_slip=(feet_sensor_velocity.square().sum(-1)*(gait==1)).sum(-1),foot_balance=balance,
        straight_knee=(.1-joint_pos[:,[3,9]]).clamp_min(0).sum(-1),foot_far=spread,
        joint_limits=limits.sum(-1),joint_torque=actuator_force.square().sum(-1),
        smoothness_joint=(.01*joint_vel.square()+((last_joint_vel-joint_vel)/dt).square()).sum(-1),
        smoothness_action=smooth.sum(-1))
    for name,section,tau in (('head',slice(0,1),.5),('feet',slice(3,5),.3),('hands',slice(5,7),.5)):
        crossing=crossed[:,section]
        if name=='feet':crossing=crossing|(gait==1)
        # Standing is no longer merged into the goal bonus; it is priced separately so a
        # no-command step cannot collect the full goal reward for doing nothing.
        idle=(move[:,None]<.5)&~crossing
        rewards[name+'gf']=gf_reward(gf[:,section],velocities[:,section],sdf[:,section],crossing,
            tau=tau,standing=idle,standing_value=standing_gf)
    for name,section in (('head',slice(0,1)),('feet',slice(3,5)),('hands',slice(5,7)),('knees',slice(7,9)),('shlds',slice(9,11))):
        rewards[name+'df']=sdf_reward(sdf[:,section],sdf_knee)
    if heading_sdf is not None:
        rewards['heading_align']=heading_align_reward(direction,pelvis_rpy[:,2],move,heading_sdf,
            margin_low=heading_margins[0],margin_high=heading_margins[1])
    return {k:torch.where(torch.isnan(v),0.,v) for k,v in rewards.items()}


def protected_hand_sdf_reward(sdf, context, knee=None):
    """Relax positive-clearance comfort shaping only in protected hand zones.

    Preserve original penalty at/below contact, fade the discretionary margin
    to zero at >=5 cm clearance. Collision checks are unchanged.
    """
    legacy_knee = torch.full_like(sdf, .05) if knee is None else knee[:,None,None].expand_as(sdf)
    active = context['hand_active'] & (context['role']==1)[:,None]
    phase = context['phase_weight'].clamp(0,1)
    blend = phase.square()*(3-2*phase)
    u = (sdf/.05).clamp(0,1)
    clearance_blend = u.square()*(3-2*u)
    effective = legacy_knee*(1-blend[:,None,None]*active[:,:,None]*clearance_blend)
    return (-20.*F.softplus((effective-sdf)/.02)).flatten(1).mean(-1)
