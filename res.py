r"""
SANS Resolution Simulator
=========================

Propagate a neutron from an isotropic source through a source and sample
pinhole and onto a detector.  For each pixel on the detector, compute the
effective resolution.

Usage
=====

Modify instrument geometry, target pixels and number of neutrons at the
bottom of the script and run using::

    python res.py

You can look at the various stages of the simulation by uncommenting
the intermediate "plot" function calls.

Theory
======

The first step is to generate a set of neutrons at small angle $\theta$
along the direction of the beam, and uniform phi, with starting position
$(x,y)$ in the source aperture.  Each neutron is given an independent
wavelength $\lambda$ from a triangular distribution resulting from the
upstream velocity selector.  The $\theta$ range is determined by the
distance between the source aperture and the sample aperture, with
extra radius to account for finite aperture size and gravity effects.

The $(\theta, \phi)$ spherical coordinates used to generate the initial
neutron population are converted to $(\theta_{\rm az}, \theta_{\rm el})
to make gravity effects easier to calculate.

The sample aperture is shifted slightly upward from $(0,0)$ so that
a beam of neutrons of the alignment wavelength will be centered on
the detector.  The aperture is not readjusted when changing wavelengths,
which will result in a main beam that is slightly above $(0,0)$ for
shorter wavelengths, or below for longer wavelengths.  At 14m, changing
from 8 A to 16 A will drop the beam by 10 pixels or so.  Since data
reduction will recenter $(q_x,q_y)$ on the detector position, the
detector is shifted so that the center pixel is at $q=0$.

After filtering through the sample aperture, we are left with a
selection neutrons at position $(s_x, s_y)$ and angle
$(s_{\rm az}, s_{\rm el})$ incident on the sample.  For each
source neutron, we generate a random position $(p_x, p_y)$ within
the target pixel on the detector, end determine the direction
$q_{\rm az}$ and throwing angle $q_{\rm el}$ required to reach that
detector position.

To determine the $(\theta,\phi)$ angle of scattering, we compare
the position $D$ on the detector of the incident neutron travelling
in a straight line without gravity (this is the beam center), to
the position $P$ on the detector of the scattered neutron travelling
in a straight line without gravity (this is the relative $(q_x,q_y$)
of the scattered neutron).   Given the position $S$ of the sample
$\theta = \tan^{-1}(||Q-D||/||D-S||)$ and
$\phi = \tan^{-1}((pn_y-d_y)/(pn_x-d_x))$.

The scattering intensity $I(q)$ which we are using to compute the
resolution effects is only a function of angle and wavelength.
The gravity effect is accounted for in determining the throwing
angle required to reach the pixel.

We can estimate the resolution of the pixel $(i,j)$ by looking
at the histograms of our populations of $\theta$, $\phi$,
$Q_\parallel = \frac{4 \pi}{\lambda} \sin(theta/2)$
and $Q_\perp = Q_\parallel (\phi - \bar\phi)$ where $\bar\phi$
is the nominal scattering angle of the pixel.

The above $(\theta,\phi)$ calculation is slightly incorrect since
the point $P$ is in the plane of the detector, which is not quite
normal to the direction of the beam $(s_{\rm az}, s_{\rm el})$
incident on the sample.  This effect is insignificant so it is
not active in the code, but it is calculated as follows.

Let $S = (s_x, s_y, z_s)$ be a point in the sample where we have a
simulated neutron travelling in direction
$n = (s_\rm{az}, s_\rm{el}) = (s_\theta \cos s_\phi, s_\theta \sin s_\phi)$,
and let $P = (p_x, p_y, z_d)$ be a point on the detector which receives the
scattered neutron.   We calculate the point $D = (d_x, d_y, z_d)
= (s_x + (z_d-z_s)*\tan s_\rm{az}, s_y + (z_d-z_s)*\tan s_\rm{el}, z_d)$
where the neutron travelling along $n$ would intercept the detector.  We then
take the plane through $D$ normal to $n$ and intersect it with the line
$\bar{SP}$ as follows:

.. math::

    Pn = S + { (D-S) \cdot n \over I \cdot n } I
       = S + { ||D-S|| \over I \cdot n } I

where the $n = (D-S) / ||D-S||$ is the plane normal to the incident neutron
where it would intercept the detector, and  $I = (P-S) / ||P-S||$ is the
direction of the scattered neutron which would intercept the detector at $P$.
Given the small angles used in SAS, $Pn \approx P$.
"""

from __future__ import division
from numpy import pi, sqrt, sin, cos, tan, arccos, arcsin, arctan2
from numpy.random import rand
import numpy as np
import matplotlib.pyplot as plt

earth_gravity = 9.80665 # m/s^2
neutron_mass = 1.00866491597 #(43) u
plancks_constant = 4.13566733e-15 #(10) eV s
electron_volt = 1.602176487e-19 #(40) J / eV
atomic_mass_constant = 1.660538782e-27 #(83) kg / u
VELOCITY_FACTOR = (plancks_constant*electron_volt
                           / (neutron_mass * atomic_mass_constant)) * 1e10
def to_velocity(wavelength): # m/s
    return VELOCITY_FACTOR / wavelength

def to_wavelength(velocity): # A
    return VELOCITY_FACTOR / velocity

def plot3(x,y,z):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)
    fig.canvas.draw()

def plot(x,y,title):
    plt.plot(x,y,'.')
    plt.axis('equal')
    plt.title(title)
    plt.grid(True)

def plot_angles(theta,phi,bins=50):
    # plot angle densities
    plt.subplot(131)
    plt.hist(theta*180/pi,bins=bins)
    plt.xlabel("theta (degrees)")
    plt.grid(True)
    plt.subplot(132)
    plt.hist(phi*180/pi,bins=bins)
    plt.xlabel("phi (degrees)")
    plt.grid(True)
    plt.subplot(133)
    plt.plot(theta*180/pi,phi*180/pi,'.')
    plt.grid(True)
    plt.xlabel('theta (degrees)')
    plt.ylabel('phi (degrees)')

def plot_q(q, phi, title, plot_phi=True):
    plt.suptitle(title)
    ax = plt.subplot(131 if plot_phi else 111)
    n, bins, patches = plt.hist(q, bins=50, normed=True)
    mean, std = np.mean(q), np.std(q, ddof=1)
    plt.plot(bins, np.exp(-0.5*((bins-mean)/std)**2)/np.sqrt(2*pi*std**2))
    q_low, q_high = mean-2.5*std, mean+3*std
    #q_low, q_high = mean**2/(mean + 3*std), mean + 3*std
    ax.vlines([q_low, q_high], 0, 1, transform=ax.get_xaxis_transform(),
              linestyle='dashed')
    plt.grid(True)
    plt.xlabel("Q (1/A)")
    if not plot_phi:
        return
    phi = np.degrees(phi)
    ax = plt.subplot(132)
    n, bins, patches = plt.hist(phi, bins=50, normed=True)
    mean, std = np.mean(phi), np.std(phi, ddof=1)
    plt.plot(bins, np.exp(-0.5*((bins-mean)/std)**2)/np.sqrt(2*pi*std**2))
    plt.grid(True)
    plt.xlabel("phi (degrees)")
    plt.subplot(133)
    plt.plot(q, phi,'.')
    plt.grid(True)
    plt.xlabel('Q (1/A)')
    plt.ylabel('phi (degrees)')

def plot_qperp(q, qperp, title):
    plt.subplot(131)
    plt.hist(q,bins=50)
    plt.grid(True)
    plt.xlabel(r"$Q_\parallel (1/A)$")
    plt.subplot(132)
    plt.hist(qperp,bins=50)
    plt.grid(True)
    plt.xlabel(r"$Q_\perp (1/A)$")
    plt.subplot(133)
    plt.plot(q, qperp,'.')
    plt.grid(True)
    plt.xlabel(r'$Q_\parallel (1/A)$')
    plt.ylabel(r'$Q_\perp (1/A)$')
    plt.suptitle(title)

def triangle(N,a=-1,b=1,c=0):
    """
    Pull random numbers from a triangular distribution over [a,b]
    with peak at c in [a,b].
    """
    cutoff = (c-a)/(b-a)
    U = rand(N)
    idx = (U>cutoff)
    X = np.empty_like(U)
    X[~idx] = a + sqrt(U[~idx]*(b-a)*(c-a))
    X[idx] = b - sqrt((1-U[idx])*(b-a)*(b-c))
    return X

def ballistics(az, el, L, x, y, d, a=earth_gravity):
    # velocity (mm/s)
    v = to_velocity(L)
    vx, vy = v*cos(el), v*sin(el)

    # distance (m) between source and sample in the direction of travel
    daz = 0.001*d/cos(az)

    # position on sample (mm)
    x = x + 1000*daz*(sin(az))
    y = y + 1000*daz*(tan(el) - 0.5*a*daz/vx**2)

    # velocity, wavelength and elevation on sample
    vy = vy - a*daz/vx
    v = sqrt(vx**2 + vy**2)
    el = arctan2(vy, vx)
    L = to_wavelength(v)
    return az, el, L, x, y

def throwing_angle(v,x,y,a=earth_gravity):
    """
    angle to throw at velocity v so that (0,0) -> (x,y)

    returns the valid index, and the plus and minus angles which
    allow the ball to get there

    if there is only one angle, it is returned twice
    """
    if a == 0:
        idx = slice(None,None,None)
        angle = arctan2(y,x)
        return idx, angle, angle
    else:
        radical = v**4 - a*(a*x**2 + 2*y*v**2)
        radical[radical<0] = 0
        #plus = arctan2(v**2 + sqrt(radical), a*x)
        minus = arctan2(v**2 - sqrt(radical), a*x)
        #plt.subplot(131); plt.hist(radical)
        #plt.subplot(132); plt.hist(plus*180/pi)
        #plt.subplot(133); plt.hist(minus*180/pi)
        return minus

def aperture_alignment(wavelength, aligned_wavelength, Dsource, Ddetector):
    # Neutrons are approximately horizontal, so they will experience a gravity
    # drop proportional to the distance travelled.  We can set the elevation
    # required to hit the target based on this distance.  We will add this
    # correction to all elevations.
    Ddetector += Dsource
    aligned_velocity = to_velocity(aligned_wavelength) # m/s
    el = 0.5*arcsin(earth_gravity*0.001*(Ddetector)/aligned_velocity**2)

    velocity = to_velocity(wavelength) # m/s
    # We need to shift the sample aperture into the ballistic trajectory by
    # computing the height of the ball at the source to sample distance
    y = Dsource*tan(el) \
        - 1000*0.5*earth_gravity*(0.001*Dsource/(velocity*cos(el)))**2

    # We need to compute the position p where the direct beam will encounter
    # the detector.
    p = Ddetector*tan(el) \
        - 1000*0.5*earth_gravity*(0.001*Ddetector/(velocity*cos(el)))**2

    return el, y, p

def nominal_q(sx, sy, az_in, el_in, az_out, el_out, dz):
    nx, ny = sx + dz * tan(az_in), sy + dz * tan(el_in)
    nd = sqrt( (nx-sx)**2 + (ny-sy)**2 + dz**2 )
    #plt.subplot(131); plot(nx/5,ny/5,'G: direct flight beam center')

    px, py = sx + dz * tan(az_out), sy + dz * tan(el_out)
    pd = sqrt( (px-sx)**2 + (py-sy)**2 + dz**2 )
    #plt.subplot(122); plot(px/5,py/5,'G: direct flight scattered beam')

    if 0:
        # Correction to move px,py into the q normal plane.  This is
        # insignificant for small angle scattering.
        nx_hat, ny_hat, nz_hat = (nx-sx)/nd, (ny-sy)/nd, dz/nd
        px_hat, py_hat, pz_hat = (px-sx)/pd, (py-sy)/pd, dz/pd
        d = nd / (px_hat*nx_hat + py_hat*ny_hat + pz_hat*nz_hat)
        px, py = sx + px_hat*d, sy + py_hat*d
    #plt.subplot(122); plot((px)/5,(py)/5,'G: scattered beam on q normal plane')

    # Note: px,py is the location of the scattered neutron relative to the
    # beam center without gravity in detector coordinates, not the qx,qy vector
    # in inverse coordinates.  This allows us to compute the scattered angle at
    # the sample, returning theta and phi.
    qd = sqrt((px-nx)**2 + (py-ny)**2)
    theta, phi = arctan2(qd, nd)/2, arctan2(py-ny, px-nx)

    return theta, phi

def resolution(R1,R2,D1,D2,dx,dy,L,dL):
    dQx = sqrt( (2*pi/(L*D2))**2 )

# All lengths are in millimeters
def pinhole(pixel_i, pixel_j, pixel_width=5, pixel_height=5,
            source_aperture=50.8, sample_aperture=12.7,
            source_distance=8500, detector_distance=4000,
            beamstop=50.8,
            wavelength=8, wavelength_resolution=0.12, aligned_wavelength=8,
            N=5000, phi_mask=7.1):
    Rsource = source_aperture/2
    Rsample = sample_aperture/2
    Rbeamstop = beamstop/2
    Dsource = source_distance
    Ddetector = detector_distance

    stats = []
    PI, PJ = np.meshgrid(pixel_i, pixel_j)
    current_j = 1000001 # arbitrary unlikely number
    for p_i, p_j in zip(PI.flat, PJ.flat):
        if current_j != p_j:
            print "j=%d"%p_j
            current_j = p_j

        # ===== Random initial state ====
        # theta is pi/2 minus latitude,  phi is longitude,  z is the rotational axis
        # The z-axis connects the center of the apertures and detector.
        # Limit the source angles to those that can make it from one side of the
        # source aperture to the other side of the sample aperture.
        #print("v_min=%.2f m/s,  gravity_drop=%.2f mm"%(min_velocity,gravity_drop))
        limit = 1 - 1/sqrt(1+((Rsource+Rsample)/Dsource)**2)
        theta, phi = arccos(1-rand(N)*limit), 2*pi*rand(N)
        #plot3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)); return
        #print "limit",limit
        #plot(theta*180/pi,phi*180/pi,"polar vs equatorial angles"); return

        # source wavelength is a triangular distribution with fwhm resolution dL/L
        L = triangle(len(theta))*wavelength*wavelength_resolution + wavelength
        #plt.hist(L,bins=50); return

        # source position: x,y is the isotropic incident location
        alpha,r = 2*pi*rand(N), Rsource*arccos(rand(N))*2/pi
        x,y = r*cos(alpha), r*sin(alpha)
        #plot(x,y,"neutron position in source aperture"); return

        # ==== Gravity correction ====
        # gravity calculations work better with azimuth and elevation
        az, el = theta*cos(phi), theta*sin(phi)

        delta_el, delta_y, delta_p \
                = aperture_alignment(wavelength, aligned_wavelength,
                                               Dsource, Ddetector)
        el += delta_el

        # ===== Compute image on sample =====
        #plot(az*180/pi,el*180/pi,"azimuthal angle vs elevation"); return
        s_az,s_el,s_L,s_x,s_y = ballistics(az,el,L,x,y,Dsource)
        #plt.hist(s_L,bins=50); return # updated wavelengths
        #plot(s_x,s_y,"G: neutron position on sample aperture"); return
        #plot(s_az,s_el,"G: sample aperture azimuthal angle vs elevation"); return

        # filter through sample aperture
        idx = (s_x**2 + (s_y-delta_y)**2 < Rsample**2)
        s_az,s_el,s_L,s_x,s_y = [w[idx] for w in (s_az,s_el,s_L,s_x,s_y)]
        #plt.hist(s_L,bins=50); plt.title("G: sample wavelength"); return
        #plot(az[idx],el[idx],"G: sample azimuthal angle vs elevation"); return
        #plot(s_az,s_el,"G: sample azimuthal angle vs elevation"); return
        #plot(s_x,s_y,"G: neutron position in sample"); return

        # ==== Compute image on detector without sample ====
        #
        #d_az, d_el, d_L, d_x, d_y = ballistics(s_az, s_el, s_L, s_x, s_y, Ddetector)

        ### filter through beam stop
        ##idx = (d_x**2 + (d_y-delta_y)**2 < Rbeamstop**2)
        ##s_az,s_el,s_L,s_x,s_y = [w[idx] for w in (s_az,s_el,s_L,s_x,s_y)]
        ##d_az,d_el,d_L,d_x,d_y = [w[idx] for w in (d_az,d_el,d_L,d_x,d_y)]

        #plot(d_x/pixel_width,d_y/pixel_height,"G: neutron detector pixel"); return


        # ==== Compute scattering theta, phi for pixel ====
        # find random point in pixel i,j to scatter to
        xl,xu = (p_i-0.5)*pixel_width, (p_i+0.5)*pixel_width
        yl,yu = delta_p+(p_j-0.5)*pixel_height, delta_p+(p_j+0.5)*pixel_height
        p_x,p_y = rand(len(s_x))*(xu-xl)+xl, rand(len(s_x))*(yu-yl)+yl
        #plot(px,py,"px,py pixel locations"); return

        # find the scattering angle necessary to reach point P on the detector
        q_az = arctan2(p_x-s_x, np.ones_like(s_x)*Ddetector)
        q_el = throwing_angle(to_velocity(s_L), 0.001*Ddetector/cos(q_az),
                              0.001*(p_y-s_y))
        #q_theta = arccos(sin(s_el)*sin(q_el) + cos(s_el)*cos(q_el)*cos(q_az-s_az))
        #q_theta_2 = arctan2(sqrt((d_x-p_x)**2+(d_y-p_y)**2)), Ddetector)
        #q_phi = arctan2(q_el,q_az)

        # Note that q scattering calculations look at positions on the detector
        # assuming neutrons travel in a straight line, and not the positions
        # according to ballistics.  The ballistics are taken into account by the
        # choice of initial angle such that the neutron will hit the target
        # position.  The scattering function operates solely on incident and
        # scattered angle with no hint of gravity, and so the resolution
        # function which mixes the scattering theory must reflect this.
        q_theta, q_phi = nominal_q(s_x, s_y, s_az, s_el, q_az, q_el, Ddetector)
        q = 4*pi*sin(q_theta)/s_L
        #return

        # filter through beam stop
        idx = (p_x**2 + p_y**2 > Rbeamstop**2)
        q_theta, q_phi, q = [w[idx] for w in (q_theta, q_phi, q)]

        # ==== calculate stats ====
        cx, cy = p_i*pixel_width, p_j*pixel_height
        theta_nominal = arctan2(sqrt(cx**2+cy**2),Ddetector)/2*180/pi
        phi_nominal = arctan2(cy,cx)*180/pi
        q_nominal = 4*pi*sin(theta_nominal*pi/180)/wavelength
        qperp_nominal = 0

        # Approximate q_perp as arc length between nominal phi and actual phi
        # at radius q.
        qperp = q*(q_phi-phi_nominal)

        if len(q) > 1:
            theta_mean, theta_std = np.mean(q_theta)*180/pi, np.std(q_theta)*180/pi
            phi_mean, phi_std = np.mean(q_phi)*180/pi, np.std(q_phi)*180/pi
            q_mean, q_std = np.mean(q), np.std(q)
            qperp_mean, qperp_std = np.mean(qperp), np.std(qperp)
            stats.append([
                theta_nominal, theta_mean, theta_std,
                phi_nominal, phi_mean, phi_std,
                q_nominal, q_mean, q_std,
                qperp_nominal, qperp_mean, qperp_std,
                ])

    config = "src-ap:%.1fcm samp-ap:%.1fcm src-dist:%.1fm det-dist:%.1fm L:%.1fA" % (
            source_aperture/10, sample_aperture/10,
            Dsource/1000, Ddetector/1000, wavelength)
    if len(stats) == 0:
        pass  # No samples fell in detector region
    elif len(stats) == 1:
        # print stats
        pixel_config = "%s pixel:%d,%d (%dX%d mm^2)" %(
              config, p_i, p_j, pixel_width, pixel_height)
        print pixel_config
        print "G nominal theta: %.2f   actual theta: %.2f +/- %.2f" \
            %(theta_nominal,theta_mean,theta_std)
        print "G nominal phi: %.2f   actual phi: %.2f +/- %.2f" \
            %(phi_nominal,phi_mean,phi_std)
        print "G nominal q: %.3f  actual q: %.3f +/- %.3f" \
            %(q_nominal, q_mean, q_std)

        #plt.hist(q_az*180/pi,bins=50); plt.title("G: scattered rotation"); plt.figure()
        #plt.hist(q_el*180/pi,bins=50); plt.title("G: scattered elevation"); return
        #plt.hist(q_theta*180/pi,bins=50); plt.title("G: Q theta"); return
        #plt.hist(q,bins=50,normed=True); plt.title("G: Q"); return

        # plot resolution
        qual = "for pixel %d,%d"%(p_i, p_j)
        #plot_angles(q_theta, q_phi)
        plot_q(q, q_phi, "Q %s"%qual, plot_phi=False)
        #plot_q(np.log10(q), q_phi, "Q %s"%qual, plot_phi=False)
        #plot_qperp(q, qperp, "Q %s"%qual)
        plt.suptitle(pixel_config)
    elif len(pixel_i) == 1 or len(pixel_j) == 1:
        stats = np.array(stats)
        plt.suptitle(config)
        plt.subplot(221)
        plt.plot(stats[:,6], stats[:,2], '.')
        plt.grid(True)
        plt.xlabel(r'$Q (1/A)$')
        plt.ylabel(r'$\Delta\theta$')
        plt.subplot(222)
        plt.plot(stats[:,6], stats[:,5], '.')
        plt.grid(True)
        plt.xlabel(r'$Q (1/A)$')
        plt.ylabel(r'$\Delta\phi$')
        plt.subplot(223)
        plt.plot(stats[:,6], stats[:,8], '.')
        plt.grid(True)
        plt.xlabel(r'$Q (1/A)$')
        plt.ylabel(r'$\Delta Q_\parallel$')
        plt.subplot(224)
        plt.plot(stats[:,6], stats[:,11], '.')
        plt.grid(True)
        plt.xlabel(r'$Q (1/A)$')
        plt.ylabel(r'$\Delta Q_\perp$')
    else:
        stats = np.array(stats)
        plt.suptitle(config)
        plt.subplot(131)
        data,title = stats[:,2], r"$\Delta\theta$"
        mask =  (PI**2+PJ**2<phi_mask**2)
        data = np.ma.array(data, mask=mask)
        #data,title = stats[:,1]-stats[:,0], r"$\theta - \hat\theta$"
        #data = np.clip(stats[:,1]-stats[:,0], 0, 0.02)
        plt.pcolormesh(pixel_i, pixel_j, data.reshape(len(pixel_i),len(pixel_j)))
        plt.grid(True)
        plt.axis('equal')
        plt.title(title)
        plt.colorbar()
        plt.subplot(132)
        data,title = stats[:,5], r"$\Delta\phi$"
        #data,title = stats[:,4]-stats[:,3], r"$\phi - \hat\phi$"
        mask =  (PI<phi_mask) & (abs(PJ)<phi_mask)
        data = np.ma.array(data, mask=mask)
        plt.pcolormesh(pixel_i, pixel_j, data.reshape(len(pixel_i),len(pixel_j)))
        plt.grid(True)
        plt.axis('equal')
        plt.title(title)
        plt.colorbar()
        plt.subplot(133)
        #data,title = stats[:,8], r"$\Delta q$"
        data,title = stats[:,8]/stats[:,6], r"$\Delta q/q$"
        mask =  (PI**2+PJ**2<phi_mask**2)
        data = np.ma.array(data, mask=mask)
        #data,title = stats[:,7]-stats[:,6], r"$q - \hat q$"
        #data = np.clip(stats[:,7]-stats[:,6], 0, 0.0005)
        plt.pcolormesh(pixel_i, pixel_j, data.reshape(len(pixel_i),len(pixel_j)))
        plt.grid(True)
        plt.axis('equal')
        plt.title(title)
        plt.colorbar()


def sphere(L, theta, phi, radius, contrast):
    q = 4*pi*sin(phi)/L
    qr = q*radius;
    bes = 1.0*np.ones_like(qr)
    idx = qr != 0
    qr = qr[idx]
    sn, cn = sin(qr), cos(qr)
    bes[idx] = 3.0*(sn-qr*cn)/(qr*qr*qr)
    fq = (bes * contrast * 4/3*pi*radius**3)
    Iq = 1e-4*fq**2
    return Iq

if __name__ == "__main__":
    # ==== select Q range
    fields = ("source_distance","detector_distance",
              "source_aperture","sample_aperture",
              "beamstop",
              "wavelength","wavelength_resolution")
    values = (
        #16270,13170, 28.6,25.4,50.8,13,0.109  # 13m @ 13A max resolution
        #15727,14547, 76.0,25.4,50.8, 6,0.124  # 14.5m @ 6A low Q
        6959, 4000,50.8,9.5,50.8, 6,0.145  # 4m @ 6A on NG7
        #13125, 13000, 50.8, 9.5, 101.6, 6, 0.14  # 13m @ 6A on NG7
        #10070, 4050,100.0,25.4,50.8, 8,0.125  # 4m @ 8A
        # 3870, 1380,100.0,25.4,50.8, 6,0.236  # 1.3m @ 6A max flux
    )
    # Parameters from NCNR VAX format files
    #   resolution.ap12dis*1000, det.dis*1000
    #   resolution.ap1, resolution.ap2
    #   det.bstop
    #   resolution.lmda, resolution.dlmda
    geom = dict(zip(fields,values))

    # ==== remove gravity
    #geom["aligned_wavelength"] = geom["wavelength"] = 0.001

    # ==== select precision
    #N = 10000000 # high precision
    N = 1000000 # mid precision
    #N = 100000 # low precision

    # ==== select detector portion
    if 0:
        # various detector regions
        #i=j=np.arange(-63.5,64) # full detector SLOW!!!
        #i=j=np.arange(-63.5,64,4) # down sampled
        i,j = np.arange(6, 64), [0] # horizontal line
        #i,j = [0], np.arange(3.5, 64) # vertical line
        #i,j = [6],[6]  # low Q point
        #i,j = [45],[45]  # high Q point
        plt.figure(); pinhole(i,j,N=N,**geom)
    else:
        # variety of single point distributions
        #geom['beamstop'] = 0.
        #plt.figure(); pinhole([0],[0],N=N,**geom)
        #plt.figure(); pinhole([1],[0],N=N,**geom)
        #plt.figure(); pinhole([2],[0],N=N,**geom)
        #plt.figure(); pinhole([3],[0],N=N,**geom)
        #plt.figure(); pinhole([4],[0],N=N,**geom)
        plt.figure(); pinhole([6],[0],N=N,**geom)
        #plt.figure(); pinhole([10],[0],N=N,**geom)
        plt.figure(); pinhole([20],[0],N=N,**geom)
        #plt.figure(); pinhole([40],[0],N=N,**geom)
        plt.figure(); pinhole([60],[0],N=N,**geom)
        #plt.figure(); pinhole([0],[60],N=N,**geom)
        #plt.figure(); pinhole([0],[-60],N=N,**geom)

    plt.show()
