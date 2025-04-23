class QRC_UNet(nn.Module):
    def __init__(self):
        super(QRC_UNet, self).__init__()
        self.encoder = timm.create_model('mobilevit_xxs', pretrained=True, features_only=True)
        enc_channels = self.encoder.feature_info.channels()  # e.g., [16, 24, 48, 64, 320]

        self.qfc = QuantumFourierConv(enc_channels[-1])
        self.rescaps = ResidualCapsuleBlock(enc_channels[-1], 256)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.adsc1 = ADSCBlock(128 + enc_channels[3], 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.adsc2 = ADSCBlock(64 + enc_channels[2], 64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.adsc3 = ADSCBlock(32 + enc_channels[1], 32)

        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.adsc4 = ADSCBlock(16 + enc_channels[0], 16)

        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        x = self.qfc(e5)
        x = self.rescaps(x)

        x = self.up1(x)
        x = self.adsc1(torch.cat([x, e4], dim=1))

        x = self.up2(x)
        x = self.adsc2(torch.cat([x, e3], dim=1))

        x = self.up3(x)
        x = self.adsc3(torch.cat([x, e2], dim=1))

        x = self.up4(x)
        x = self.adsc4(torch.cat([x, e1], dim=1))

        x = self.final_conv(x)
        return torch.sigmoid(x)
