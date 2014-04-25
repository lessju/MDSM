from math import degrees, radians
from datetime import datetime
import ephem
import math

day = str(datetime.now()) # day = '1998/8/10 23:10:00'
longitude = ephem.degrees('11.64595')
latitude = ephem.degrees('44.524')


def juldate2ephem(num):
    """Convert Julian date to ephem date, measured from noon, Dec. 31, 1899."""
    return ephem.date(num - 2415020.)

def ephem2juldate(num):
    """Convert ephem date (measured from noon, Dec. 31, 1899) to Julian date."""
    return float(num + 2415020.)

def topoToEq(az, el):
    """ Convert from topocentric to Equatorial coords
        Input in degrees """
    
    observer = ephem.Observer()
    observer.lon = longitude
    observer.lat = latitude
    observer.date = juldate2ephem(ephem.julian_date())

    print 'Time', observer.date
    print 'AZ/EL (degrees)', float(ephem.degrees(az)), float(ephem.degrees(el))

    az = radians(az)
    el = radians(az)

    print 'RA/DEC (degrees)', observer.radec_of(az, el)[0], observer.radec_of(az, el)[1]
    print

def eqToTopo(ra, dec):
    """ Convert from equatorial to topocentric coords
        Inputs in hms and dms """

    star = ephem.FixedBody()
    star._ra = ra
    star._dec = dec

    observer = ephem.Observer()
    observer.date = juldate2ephem(ephem.julian_date())
    observer.lon = longitude
    observer.lat = latitude

    star.compute(observer)

    print 'Time', observer.date
    print 'RA/DEC (degrees)', star.ra, star.dec
    print 'AZ/EL (degrees)', star.az, star.alt
    print


if __name__ == "__main__":

    # B0329
    eqToTopo('03:32:59.368', '+54:34:43.57')

    # Example (az and el in radians)
    topoToEq(degrees(180), degrees(46.1837))
