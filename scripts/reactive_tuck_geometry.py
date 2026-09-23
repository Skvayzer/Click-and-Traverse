"""CPU FK and conservative proxy constraints for measurement-only tuck search."""
import json,xml.etree.ElementTree as ET
import numpy as np
import mujoco
from cat_mjlab.model import assemble_training_xml
from cat_mjlab.constants import DEFAULT_QPOS,HAND_SITES
from cat_mjlab.collision import PROPOSAL

DIRECTIONS={name:np.array(v,dtype=float) for name,v in [
 ('front',(1,0,0)),('front_left',(1,1,0)),('left',(0,1,0)),('back_left',(-1,1,0)),
 ('back',(-1,0,0)),('back_right',(-1,-1,0)),('right',(0,-1,0)),('front_right',(1,-1,0)),
 ('above',(0,0,1)),('below',(0,0,-1))]}
DIRECTIONS={k:v/np.linalg.norm(v) for k,v in DIRECTIONS.items()}
BUCKETS={'danger':(.03,.08),'anticipation':(.09,.20),'negative':(.20,.35)}


class Geometry:
    def __init__(self):
        root=ET.fromstring(assemble_training_xml());bodies={b.get('name'):b for b in root.findall('.//body')}
        self.shapes=json.loads(PROPOSAL.read_text())['shapes']
        for i,s in enumerate(self.shapes):
            attrs=dict(name=f'audit_proxy_{i}',type=s['kind'],contype='0',conaffinity='0',density='0',group='5')
            nums=lambda x:' '.join(str(v) for v in x)
            if s['kind']=='capsule':attrs.update(fromto=nums(np.asarray(s['endpoints']).ravel()),size=str(s['radius']))
            else:
                attrs.update(pos=nums(s['center']),quat=nums(s['quat']),size=nums(s['half_size']) if s['kind']=='box' else str(s['radius']))
            ET.SubElement(bodies[s['body_name']],'geom',**attrs)
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));self.data=mujoco.MjData(self.model)
        self.ids=np.array([self.model.geom(f'audit_proxy_{i}').id for i in range(len(self.shapes))])
        self.q=np.array(DEFAULT_QPOS,dtype=float);self.q[2]=.8
        self.nominal=self.q[19:].copy()
        limits=self.model.jnt_range[13:];mid=limits.mean(1);span=limits[:,1]-limits[:,0]
        self.lo=np.maximum(self.nominal-.8,mid-.475*span)
        self.hi=np.minimum(self.nominal+.8,mid+.475*span)
        self.hand_index=next(i for i,s in enumerate(self.shapes) if s['id']=='left_hand')
        self.hand_radius=self.shapes[self.hand_index]['radius']
        self.upper=np.array([s['group'] in ('arms','hands','trunk','head') for s in self.shapes])
        self.arm=np.array([s['group']=='arms' for s in self.shapes])
        self.fixed=np.array([s['group'] in ('feet','legs') or s['id']=='pelvis' for s in self.shapes])
        self.self_pairs=[];self.excluded=[]
        body_ids=self.model.geom_bodyid[self.ids]
        def ancestors(body):
            a=[int(body)]
            while a[-1]>0:a.append(int(self.model.body_parentid[a[-1]]))
            return a
        ancestors_all=[ancestors(b) for b in body_ids]
        for i in range(len(self.ids)):
            for j in range(i):
                if not (self.upper[i] or self.upper[j]):continue
                ai,aj=ancestors_all[i],ancestors_all[j]
                common=next(b for b in ai if b in aj);edges=ai.index(common)+aj.index(common)
                if edges<=2:
                    self.excluded.append((i,j));continue
                self.self_pairs.append((i,j))
        self.self_pairs=np.array(self.self_pairs,dtype=int)
        self.set(self.nominal);self.start_hand=self.hand().copy()
        initial=self.self_distances()
        self.self_margins=np.minimum(initial*.5,.002)
        self.baseline_overlaps=[(self.shapes[i]['id'],self.shapes[j]['id'],float(d)) for (i,j),d in zip(self.self_pairs,initial) if d<0]
        # Fixed transforms of all approved primitives, exact world geometry.
        self.set(self.nominal)
        # Conservative displacement bound per shape per upper joint. Chain body
        # offsets + joint anchors + local enclosing radius bound all rotations.
        self.levers=np.zeros((len(self.shapes),17))
        for i,(s,b) in enumerate(zip(self.shapes,body_ids)):
            extent=(np.linalg.norm(s['center'])+(np.linalg.norm(s['half_size']) if s['kind']=='box' else s['radius']))
            if s['kind']=='capsule':extent=max(np.linalg.norm(s['endpoints'],axis=1))+s['radius']
            for k,joint in enumerate(range(13,self.model.njnt)):
                ancestor=int(self.model.jnt_bodyid[joint]);chain=ancestors_all[i]
                if ancestor not in chain:continue
                length=extent+np.linalg.norm(self.model.jnt_pos[joint])
                cur=int(b)
                while cur!=ancestor:
                    length+=np.linalg.norm(self.model.body_pos[cur]);cur=int(self.model.body_parentid[cur])
                self.levers[i,k]=length

    def set(self,x):
        self.data.qpos[:]=self.q;self.data.qpos[19:]=x
        mujoco.mj_kinematics(self.model,self.data)

    def hand(self):return self.data.geom_xpos[self.ids[self.hand_index]]

    def self_distances(self):
        """Vectorized analytic proxy separations, independent of GJK tolerances."""
        p=self.data.geom_xpos[self.ids];r=self.data.geom_xmat[self.ids].reshape(-1,3,3)
        kinds=np.array([s['kind'] for s in self.shapes]);rad=np.array([s.get('radius',0.) for s in self.shapes])
        half=np.array([np.linalg.norm(np.asarray(s['endpoints'])[1]-s['endpoints'][0])/2 if s['kind']=='capsule' else 0. for s in self.shapes])
        size=np.array([s.get('half_size',[0,0,0]) for s in self.shapes])
        a=p-r[:,:,2]*half[:,None];b=p+r[:,:,2]*half[:,None]
        ii,jj=self.self_pairs.T;out=np.empty(len(ii))
        caps=(kinds[ii]!='box')&(kinds[jj]!='box')
        if caps.any():
            i,j=ii[caps],jj[caps];u=b[i]-a[i];v=b[j]-a[j];w=a[i]-a[j]
            A=(u*u).sum(-1);B=(u*v).sum(-1);C=(v*v).sum(-1);D=(u*w).sum(-1);E=(v*w).sum(-1)
            den=A*C-B*B
            s=(B*E-C*D)/np.maximum(den,1e-30);t=(A*E-B*D)/np.maximum(den,1e-30)
            ds=np.linalg.norm(w+s[:,None]*u-t[:,None]*v,axis=-1)
            ds=np.where((den>1e-20)&(s>=0)&(s<=1)&(t>=0)&(t<=1),ds,np.inf)
            for point in (a[i],b[i]):
                t=np.clip(((point-a[j])*v).sum(-1)/np.maximum(C,1e-30),0,1)
                ds=np.minimum(ds,np.linalg.norm(point-a[j]-t[:,None]*v,axis=-1))
            for point in (a[j],b[j]):
                t=np.clip(((point-a[i])*u).sum(-1)/np.maximum(A,1e-30),0,1)
                ds=np.minimum(ds,np.linalg.norm(point-a[i]-t[:,None]*u,axis=-1))
            out[caps]=ds-rad[i]-rad[j]
        mixed=(kinds[ii]=='box')^(kinds[jj]=='box')
        if mixed.any():
            i=np.where(kinds[ii[mixed]]=='box',jj[mixed],ii[mixed]);j=np.where(kinds[ii[mixed]]=='box',ii[mixed],jj[mixed])
            start=np.einsum('nji,nj->ni',r[j],a[i]-p[j]);end=np.einsum('nji,nj->ni',r[j],b[i]-p[j]);h=size[j];v=end-start
            divisor=np.where(v!=0,v,1.)
            crossing=np.concatenate(((-h-start)/divisor,(h-start)/divisor),-1)
            crossing=np.where(np.concatenate((v!=0,v!=0),-1),crossing,0).clip(0,1)
            boundaries=np.sort(np.c_[np.zeros(len(i)),np.ones(len(i)),crossing],axis=1)
            lo,hi=boundaries[:,:-1],boundaries[:,1:];mid=(lo+hi)/2
            pos=start[:,None]+mid[:,:,None]*v[:,None];active=np.abs(pos)>h[:,None]
            offset=start[:,None]-np.where(pos>=0,1.,-1.)*h[:,None]
            quad=np.where(active,v[:,None]**2,0).sum(-1);lin=np.where(active,v[:,None]*offset,0).sum(-1)
            t=np.clip(np.where(quad>0,-lin/np.maximum(quad,1e-30),mid),lo,hi)
            q=np.maximum(np.abs(start[:,None]+t[:,:,None]*v[:,None])-h[:,None],0)
            out[mixed]=np.sqrt((q*q).sum(-1).min(-1))-rad[i]
        boxes=(kinds[ii]=='box')&(kinds[jj]=='box')
        if boxes.any():
            i,j=ii[boxes],jj[boxes];delta=p[j]-p[i]
            axes=np.concatenate((r[i].transpose(0,2,1),r[j].transpose(0,2,1),
                np.cross(r[i].transpose(0,2,1)[:,:,None],r[j].transpose(0,2,1)[:,None]).reshape(-1,9,3)),axis=1)
            norm=np.linalg.norm(axes,axis=-1);axes=axes/np.maximum(norm[...,None],1e-30)
            ai=np.einsum('nai,nij->naj',axes,r[i]);aj=np.einsum('nai,nij->naj',axes,r[j])
            sep=np.abs(np.einsum('nai,ni->na',axes,delta))-(np.abs(ai)*size[i,None]).sum(-1)-(np.abs(aj)*size[j,None]).sum(-1)
            out[boxes]=np.where(norm>1e-7,sep,-np.inf).max(-1)
        return out

    def object_distances(self,center):
        p=self.data.geom_xpos[self.ids];r=self.data.geom_xmat[self.ids].reshape(-1,3,3)
        local=np.einsum('nji,nj->ni',r,center-p);out=[]
        for i,s in enumerate(self.shapes):
            if s['kind']=='box':
                offset=np.abs(local[i])-np.asarray(s['half_size'])
                d=np.linalg.norm(np.maximum(offset,0))+min(max(offset),0)
            elif s['kind']=='capsule':
                half=np.linalg.norm(np.asarray(s['endpoints'])[1]-s['endpoints'][0])/2
                delta=local[i].copy();delta[2]-=np.clip(delta[2],-half,half)
                d=np.linalg.norm(delta)-s['radius']
            else:d=np.linalg.norm(local[i])-s['radius']
            out.append(d-.06)
        return np.array(out)

    def floor_distances(self):
        p=self.data.geom_xpos[self.ids];r=self.data.geom_xmat[self.ids].reshape(-1,3,3);out=[]
        for i,s in enumerate(self.shapes):
            if s['group']=='feet':out.append(1.);continue
            if s['kind']=='box':support=np.abs(r[i,2])@np.asarray(s['half_size'])
            elif s['kind']=='capsule':support=abs(r[i,2,2])*np.linalg.norm(np.asarray(s['endpoints'])[1]-s['endpoints'][0])/2+s['radius']
            else:support=s['radius']
            out.append(p[i,2]-support)
        return np.array(out)

    def trajectory(self,direction,gap,speed):
        v=DIRECTIONS[direction];end=self.start_hand+v*(self.hand_radius+.06+gap)
        start=self.start_hand+v*(self.hand_radius+.06+.35)
        if direction=='below':start[2]=.062
        return start,end,float(np.linalg.norm(end-start)/speed)

    def certificate(self,target,start,end,duration,*,margin=.001,max_depth=17):
        """Continuous fixed-base straight-joint-path certificate, not sample-only.

        At an interval midpoint, each signed clearance is reduced by a valid
        worst-case rigid-shape displacement bound to either end. Subdivide
        ambiguous intervals. Report uncertified rather than infer safety.
        """
        delta=target-self.nominal;move=self.levers@np.abs(delta);obj=np.linalg.norm(end-start)
        li=self.levers[self.self_pairs[:,0]];lj=self.levers[self.self_pairs[:,1]]
        pair_bound=((li+lj)*((li>0)^(lj>0)))@np.abs(delta)
        speed_time=np.max(np.abs(delta))/2
        minima=dict(object=1e6,self=1e6,floor=1e6);count=0;failed=[]
        def check(a,b,depth):
            nonlocal count
            t=(a+b)/2;self.set(self.nominal+t*delta)
            d=self.object_distances(start+t*(end-start));sd=self.self_distances();fd=self.floor_distances();count+=1
            for k,val in [('object',d.min()),('self',sd.min()),('floor',fd.min())]:minima[k]=min(minima[k],float(val))
            if d.min()<margin or fd.min()<margin or sd.min()<1e-6:
                failed.append(float(t));return False
            half=(b-a)/2
            if (d-(move+obj)*half).min()>=margin and (fd-move*half).min()>=margin and (sd-pair_bound*half).min()>=1e-6:return True
            if depth>=max_depth:
                failed.append(float(t));return False
            return check(a,t,depth+1) and check(t,b,depth+1)
        passed=speed_time<=duration+1e-8 and check(0.,1.,0)
        return dict(passed=bool(passed),failed_fractions=failed,min_sampled_clearances=minima,interval_evaluations=count,
            minimum_command_time_s=float(speed_time),allocated_time_s=duration,margin_m=margin,method='recursive Lipschitz bounds, all proxy pairs except graph-neighbors <=2 edges')
