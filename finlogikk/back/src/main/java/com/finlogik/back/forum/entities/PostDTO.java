package com.finlogik.back.forum.entities;
import lombok.*;

import java.io.Serializable;
import java.util.Date;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@ToString
public class PostDTO implements Serializable{
    private Long idPost;
    private String content;
    private Integer likes;

    private Integer dislikes;
    private Date creationDate;
    private Boolean modified;
    private String userId;
}