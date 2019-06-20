import imutils
import cv2
import numpy as np

src_path = "./test-img/"
#There are 4 games rightnow in Ludex app, so we can easily detect the winner in them by openCv
#At first we detect the computer or Tv screen from the photo, and then base of that will find the winner in different games

def order_corner_points(corners):
    # Separate corners into individual points
    # Index 0 - top-right
    #       1 - top-left
    #       2 - bottom-left
    #       3 - bottom-right
    corners = [(corner[0][0], corner[0][1]) for corner in corners]
    top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
    return (top_l, top_r, bottom_r, bottom_l)

def perspective_transform(image, corners):
    # Order points in clockwise order
    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners
    # Determine width of new image which is the max distance between
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                    [0, height - 1]], dtype = "float32")

    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")
    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)
    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height))


# img = cv2.imread(r"C:\Users\Ebi\Documents\Python\TextRecognize\test-img\img1.png")

def roi(img_path):
    image = cv2.imread(img_path)
    if  image is None :
       print ("Error loading image")

    ratio = image.shape[0] / 300.0
    image = imutils.resize(image, height=300)
    realImage = image.copy()

# convert the image to grayscale, blur it, and find edges in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None

# loop over our contours
    for c in cnts:
    # approximate the contour
         peri = cv2.arcLength(c, True)
         approx = cv2.approxPolyDP(c, 0.015 * peri, True)

         if len(approx) == 4:
             screenCnt = approx
             transformed = perspective_transform(realImage, screenCnt)
             cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
             break
         else:
            transformed= image

    cv2.imshow("transformed", transformed)
    return transformed



def winner_first_game(img_path):
    img=roi(img_path)
   # upper = np.array([255, 255, 170], dtype='uint8')
  #  lower = np.array([100, 100, 0], dtype='uint8')
    height, width, depth = img.shape
    circle_img = np.zeros((height, width), np.uint8)
    # define a circle mask
    mask = cv2.circle(circle_img, (int(width / 2) - 1, int(height / 2) - 50), 10, 1, thickness=-1)

    masked_img = cv2.bitwise_and(img, img, mask=circle_img)
    circle_locations = mask == 1

    #search if there is a yellow color in the mask,( there is two players with in colors of yellow and orange)
    bgr = img[circle_locations]
    rgb = bgr[..., ::-1]
    yellow = [255, 255, 0]
    isyellow = False
    if yellow in rgb:
        isyellow = True
    cv2.imshow("masked", masked_img)
    return isyellow



def winner_second_game(img_path):
    img=roi(img_path)

    #resize the image to width=1000
    W = 1000
    height, width, chan = img.shape
    imgScale = W / width
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
    newimg = cv2.resize(img, (int(newX), int(newY)))

    #define template
    template = cv2.imread(src_path +'template.jpg')
    w, h, c= template.shape

    #use matchtemplate to find the template
    res = cv2.matchTemplate(newimg, template, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)

    w_img, h_img , c_image= newimg.shape
    # print(w_img)
    # print(h_img)
    # print("Top left of template", maxLoc[0])


    #find the winner base of place of template detection
    if maxLoc[0] < 300 :
        winner = 'player1'
    elif maxLoc[0] > 300 and maxLoc[0] < 520:
        winner = 'player2'
    elif maxLoc[0] > 550 and maxLoc[0] < 770:
        winner = 'player3'
    elif maxLoc[0] > 800:
        winner = 'player4'



    cv2.rectangle(newimg, (maxLoc[0], maxLoc[1]),
                 (maxLoc[0] + w, maxLoc[1] + h), (0, 0, 255), 2)
    cv2.imshow("newimg/Visualize", newimg)
    return winner





def detect_winner(game_id,img ):

    if (game_id == 1):
        print("Yellow player is WINNER") if (winner_first_game(src_path + img)) else print(
            "Orange player is WINNER")
    elif (game_id == 2):
        winner = winner_second_game(src_path + img)
        print(f"{winner} is WINNER")
    cv2.waitKey(0)


#test the function
detect_winner(2,'smash4.png' )





