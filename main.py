from base.python.Protocol.AutoIp import AutoIp
from base.python.Control import Control
from image_streaming_and_storing.python.continuous_streamingTA import runContinuousStreamingTA
from image_streaming_and_storing.python.snapshotsTA import runSnapshotsDemo

if __name__ == '__main__':
    ip_address = "169.254.176.166"
    autoIp = AutoIp(ip_address)
    devices = autoIp.scan()
    for device in devices:
        print(f"Device name:  {device.deviceIdent}")
        print(f"SerialNumber: {device.serialNumber}")
        print(f"MAC Address:  {device.macAddress}")
        print(f"IP Address:   {device.ipAddress}")
        print(f"Network Mask: {device.netmask}")
        print(f"CoLa port:    {device.colaPort}")
        print(f"CoLa version: {int(device.colaVersion)}")
    print("Number of found devices: ", len(devices))

    ip_address_cam = "169.254.95.62"
    port_cam = 2122
    deviceControl = Control(ip_address_cam, 'Cola2', port_cam)
    deviceControl.open()

    name, version = deviceControl.getIdent()
    print(f"DeviceIdent: {name} {version}")

    deviceControl.login(Control.USERLEVEL_SERVICE, "CUST_SERV")
    print("\nLogin with user level SERVICE was successful")

    new_frame_period_us = 100000
    deviceControl.setFramePeriodUs(new_frame_period_us)
    print(f"Set FramePeriodUS to {new_frame_period_us}")
    frame_period_us = deviceControl.getFramePeriodUs()
    print(f"Read framePeriodUs: {frame_period_us}\n")

    runContinuousStreamingTA(
        deviceControl, ip_address=ip_address_cam, transport_protocol="UDP", receiver_ip=ip_address, streaming_port=port_cam,
        device_type="Visionary-T Mini", count=100, output_prefix='test', write_files=True
    )

    # runSnapshotsDemo(
    #     deviceControl, ip_address=ip_address_cam, transport_protocol="UDP", receiver_ip=ip_address,
    #     streaming_port=port_cam, device_type="Visionary-T Mini", number_frames=20, output_prefix='test',
    #     poll_period_ms=300, write_files=True
    # )

    # deviceControl.logout()
    deviceControl.close()
