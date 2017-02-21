using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Drawing2D;

namespace TsvTool.Utility
{
    public static class ImageUtility
    {
        static ImageCodecInfo GetEncoder(ImageFormat format)
        {
            ImageCodecInfo returncodec = null;
            ImageCodecInfo[] codecs = ImageCodecInfo.GetImageDecoders();
            foreach (var codec in codecs)
            {
                if (codec.FormatID == format.Guid)
                    returncodec = codec;
            }
            return returncodec;
        }

        public static byte[] SaveImageToJpegInBuffer(Bitmap img, Int64 quality = 90L)
        {
            var jpgEncoder = GetEncoder(ImageFormat.Jpeg);
            var myEncoder = System.Drawing.Imaging.Encoder.Quality;
            var myEncoderParas = new EncoderParameters(1);
            var myEncoderPara = new EncoderParameter(myEncoder, quality);
            myEncoderParas.Param[0] = myEncoderPara;

            using (var mw = new MemoryStream())
            {
                img.Save(mw, jpgEncoder, myEncoderParas);
                return mw.ToArray();
            }
        }

        // only down size image if image (longer side or short side) is larger than max_size
        // if limit_shorter_side == true, it will resize image to max_size based on its shorter side,
        // otherwise, resize based on its longer side
        public static Bitmap DownsizeImage(Bitmap img, int max_size, bool limit_shorter_side)
        {
            int w = img.Width, h = img.Height;
            int ori_size = limit_shorter_side ? Math.Min(w, h) : Math.Max(w, h);
            if (ori_size > max_size)
            {
                w = (int)((float)img.Width * max_size / ori_size + 0.5f);
                h = (int)((float)img.Height * max_size / ori_size + 0.5f);

                var destRect = new Rectangle(0, 0, w, h);
                var destImage = new Bitmap(w, h, PixelFormat.Format24bppRgb);

                destImage.SetResolution(img.HorizontalResolution, img.VerticalResolution);

                using (var g = Graphics.FromImage(destImage))
                {
                    g.CompositingQuality = CompositingQuality.HighQuality;
                    g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    g.DrawImage(img, destRect, 0, 0, img.Width, img.Height, GraphicsUnit.Pixel);
                }

                return destImage;
            }

            return img;
        }

    }
}
